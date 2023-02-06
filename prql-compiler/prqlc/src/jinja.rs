//! Handling of Jinja templates
//!
//! dbt is using the following pipeline: `Jinja+SQL -> SQL -> execution`.
//!
//! To prevent messing up the templates, we have create the following pipeline:
//! ```
//! Jinja+PRQL -> Jinja+SQL -> SQL -> execution
//! ```
//!
//! But because prql-compiler does not (and should not) know how to handle Jinja,
//! we have to extract the interpolations, replace them something that is valid PRQL,
//! compile the query and inject interpolations back in.
//!
//! Unfortunately, this requires parsing Jinja.
//!
//! use crate::compiler::tokens::{Span, Token};

use anyhow::Result;
use itertools::Itertools;

pub fn blah(source: &str) -> Result<()> {
    let tokens: Vec<(minijinja::Token, _)> = minijinja::tokenize(source, false).try_collect()?;

    dbg!(tokens);

    Ok(())
}

#[allow(unused)]
mod minijinja {
    //! Shamelessly stolen from https://github.com/mitsuhiko/minijinja
    //! Should I include the license?

    use utils::{memchr, memstr, unescape};

    use std::fmt;

    /// Represents a token in the stream.
    #[derive(Debug)]
    #[cfg_attr(
        feature = "unstable_machinery_serde",
        derive(serde::Serialize),
        serde(tag = "name", content = "payload")
    )]
    pub enum Token<'a> {
        /// Raw template data.
        TemplateData(&'a str),
        /// Variable block start.
        VariableStart,
        /// Variable block end
        VariableEnd,
        /// Statement block start
        BlockStart,
        /// Statement block end
        BlockEnd,
        /// An identifier.
        Ident(&'a str),
        /// A borrowed string.
        Str(&'a str),
        /// An allocated string.
        String(String),
        /// An integer (limited to i64)
        Int(i64),
        /// A float
        Float(f64),
        /// A plus (`+`) operator.
        Plus,
        /// A plus (`-`) operator.
        Minus,
        /// A mul (`*`) operator.
        Mul,
        /// A div (`/`) operator.
        Div,
        /// A floor division (`//`) operator.
        FloorDiv,
        /// Power operator (`**`).
        Pow,
        /// A mod (`%`) operator.
        Mod,
        /// The bang (`!`) operator.
        Bang,
        /// A dot operator (`.`)
        Dot,
        /// The comma operator (`,`)
        Comma,
        /// The colon operator (`:`)
        Colon,
        /// The tilde operator (`~`)
        Tilde,
        /// The assignment operator (`=`)
        Assign,
        /// The pipe symbol.
        Pipe,
        /// `==` operator
        Eq,
        /// `!=` operator
        Ne,
        /// `>` operator
        Gt,
        /// `>=` operator
        Gte,
        /// `<` operator
        Lt,
        /// `<=` operator
        Lte,
        /// Open Bracket
        BracketOpen,
        /// Close Bracket
        BracketClose,
        /// Open Parenthesis
        ParenOpen,
        /// Close Parenthesis
        ParenClose,
        /// Open Brace
        BraceOpen,
        /// Close Brace
        BraceClose,
    }

    impl<'a> fmt::Display for Token<'a> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Token::TemplateData(_) => write!(f, "template-data"),
                Token::VariableStart => write!(f, "start of variable block"),
                Token::VariableEnd => write!(f, "end of variable block"),
                Token::BlockStart => write!(f, "start of block"),
                Token::BlockEnd => write!(f, "end of block"),
                Token::Ident(_) => write!(f, "identifier"),
                Token::Str(_) | Token::String(_) => write!(f, "string"),
                Token::Int(_) => write!(f, "integer"),
                Token::Float(_) => write!(f, "float"),
                Token::Plus => write!(f, "`+`"),
                Token::Minus => write!(f, "`-`"),
                Token::Mul => write!(f, "`*`"),
                Token::Div => write!(f, "`/`"),
                Token::FloorDiv => write!(f, "`//`"),
                Token::Pow => write!(f, "`**`"),
                Token::Mod => write!(f, "`%`"),
                Token::Bang => write!(f, "`!`"),
                Token::Dot => write!(f, "`.`"),
                Token::Comma => write!(f, "`,`"),
                Token::Colon => write!(f, "`:`"),
                Token::Tilde => write!(f, "`~`"),
                Token::Assign => write!(f, "`=`"),
                Token::Pipe => write!(f, "`|`"),
                Token::Eq => write!(f, "`==`"),
                Token::Ne => write!(f, "`!=`"),
                Token::Gt => write!(f, "`>`"),
                Token::Gte => write!(f, "`>=`"),
                Token::Lt => write!(f, "`<`"),
                Token::Lte => write!(f, "`<=`"),
                Token::BracketOpen => write!(f, "`[`"),
                Token::BracketClose => write!(f, "`]`"),
                Token::ParenOpen => write!(f, "`(`"),
                Token::ParenClose => write!(f, "`)`"),
                Token::BraceOpen => write!(f, "`{{`"),
                Token::BraceClose => write!(f, "`}}`"),
            }
        }
    }

    /// Token span information
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    #[cfg_attr(feature = "unstable_machinery_serde", derive(serde::Serialize))]
    pub struct Span {
        pub start_line: usize,
        pub start_col: usize,
        pub end_line: usize,
        pub end_col: usize,
    }

    impl fmt::Debug for Span {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                " @ {}:{}-{}:{}",
                self.start_line, self.start_col, self.end_line, self.end_col
            )
        }
    }

    enum LexerState {
        Template,
        InVariable,
        InBlock,
    }

    struct TokenizerState<'s> {
        stack: Vec<LexerState>,
        rest: &'s str,
        failed: bool,
        current_line: usize,
        current_col: usize,
    }

    #[inline(always)]
    fn find_marker(a: &str) -> Option<(usize, bool)> {
        let bytes = a.as_bytes();
        let mut offset = 0;
        loop {
            let idx = match memchr(&bytes[offset..], b'{') {
                Some(idx) => idx,
                None => return None,
            };
            if let Some(b'{' | b'%' | b'#') = bytes.get(offset + idx + 1).copied() {
                return Some((
                    offset + idx,
                    bytes.get(offset + idx + 2).copied() == Some(b'-'),
                ));
            }
            offset += idx + 1;
        }
    }

    #[cfg(feature = "unicode")]
    fn lex_identifier(s: &str) -> usize {
        s.chars()
            .enumerate()
            .map_while(|(idx, c)| {
                let cont = if c == '_' {
                    true
                } else if idx == 0 {
                    unicode_ident::is_xid_start(c)
                } else {
                    unicode_ident::is_xid_continue(c)
                };
                cont.then(|| c.len_utf8())
            })
            .sum::<usize>()
    }

    #[cfg(not(feature = "unicode"))]
    fn lex_identifier(s: &str) -> usize {
        s.as_bytes()
            .iter()
            .enumerate()
            .take_while(|&(idx, &c)| {
                if c == b'_' {
                    true
                } else if idx == 0 {
                    c.is_ascii_alphabetic()
                } else {
                    c.is_ascii_alphanumeric()
                }
            })
            .count()
    }

    fn skip_basic_tag(block_str: &str, name: &str) -> Option<(usize, bool)> {
        let mut ptr = block_str;
        let mut trim = false;

        if let Some(rest) = ptr.strip_prefix('-') {
            ptr = rest;
        }
        while let Some(rest) = ptr.strip_prefix(|x: char| x.is_ascii_whitespace()) {
            ptr = rest;
        }

        ptr = match ptr.strip_prefix(name) {
            Some(ptr) => ptr,
            None => return None,
        };

        while let Some(rest) = ptr.strip_prefix(|x: char| x.is_ascii_whitespace()) {
            ptr = rest;
        }
        if let Some(rest) = ptr.strip_prefix('-') {
            ptr = rest;
            trim = true;
        }
        ptr = match ptr.strip_prefix("%}") {
            Some(ptr) => ptr,
            None => return None,
        };

        Some((block_str.len() - ptr.len(), trim))
    }

    impl<'s> TokenizerState<'s> {
        fn advance(&mut self, bytes: usize) -> &'s str {
            let (skipped, new_rest) = self.rest.split_at(bytes);
            for c in skipped.chars() {
                match c {
                    '\n' => {
                        self.current_line += 1;
                        self.current_col = 0;
                    }
                    _ => self.current_col += 1,
                }
            }
            self.rest = new_rest;
            skipped
        }

        #[inline(always)]
        fn loc(&self) -> (usize, usize) {
            (self.current_line, self.current_col)
        }

        fn span(&self, start: (usize, usize)) -> Span {
            let (start_line, start_col) = start;
            Span {
                start_line,
                start_col,
                end_line: self.current_line,
                end_col: self.current_col,
            }
        }

        fn syntax_error(&mut self, msg: &'static str) -> error::Error {
            self.failed = true;
            error::Error::new(error::ErrorKind::SyntaxError, msg)
        }

        fn eat_number(&mut self) -> Result<(Token<'s>, Span), error::Error> {
            let old_loc = self.loc();
            let mut is_float = false;
            let num_len = self
                .rest
                .as_bytes()
                .iter()
                .take_while(|&&c| {
                    if !is_float && c == b'.' {
                        is_float = true;
                        true
                    } else {
                        c.is_ascii_digit()
                    }
                })
                .count();
            let num = self.advance(num_len);
            Ok(if is_float {
                (
                    Token::Float(match num.parse::<f64>() {
                        Ok(val) => val,
                        Err(_) => return Err(self.syntax_error("invalid float")),
                    }),
                    self.span(old_loc),
                )
            } else {
                (
                    Token::Int(match num.parse::<i64>() {
                        Ok(val) => val,
                        Err(_) => return Err(self.syntax_error("invalid integer")),
                    }),
                    self.span(old_loc),
                )
            })
        }

        fn eat_identifier(&mut self) -> Result<(Token<'s>, Span), error::Error> {
            let ident_len = lex_identifier(self.rest);
            if ident_len > 0 {
                let old_loc = self.loc();
                let ident = self.advance(ident_len);
                Ok((Token::Ident(ident), self.span(old_loc)))
            } else {
                Err(self.syntax_error("unexpected character"))
            }
        }

        fn eat_string(&mut self, delim: u8) -> Result<(Token<'s>, Span), error::Error> {
            let old_loc = self.loc();
            let mut escaped = false;
            let mut has_escapes = false;
            let str_len = self
                .rest
                .as_bytes()
                .iter()
                .skip(1)
                .take_while(|&&c| match (escaped, c) {
                    (true, _) => {
                        escaped = false;
                        true
                    }
                    (_, b'\\') => {
                        escaped = true;
                        has_escapes = true;
                        true
                    }
                    (_, c) if c == delim => false,
                    _ => true,
                })
                .count();
            if escaped || self.rest.as_bytes().get(str_len + 1) != Some(&delim) {
                return Err(self.syntax_error("unexpected end of string"));
            }
            let s = self.advance(str_len + 2);
            Ok(if has_escapes {
                (
                    Token::String(match unescape(&s[1..s.len() - 1]) {
                        Ok(unescaped) => unescaped,
                        Err(err) => return Err(err),
                    }),
                    self.span(old_loc),
                )
            } else {
                (Token::Str(&s[1..s.len() - 1]), self.span(old_loc))
            })
        }

        fn skip_whitespace(&mut self) {
            let skip = self
                .rest
                .chars()
                .map_while(|c| c.is_whitespace().then(|| c.len_utf8()))
                .sum::<usize>();
            if skip > 0 {
                self.advance(skip);
            }
        }
    }

    /// Tokenizes the source.
    pub fn tokenize(
        input: &str,
        in_expr: bool,
    ) -> impl Iterator<Item = Result<(Token<'_>, Span), error::Error>> {
        let mut state = TokenizerState {
            rest: input,
            stack: vec![if in_expr {
                LexerState::InVariable
            } else {
                LexerState::Template
            }],
            failed: false,
            current_line: 1,
            current_col: 0,
        };
        let mut trim_leading_whitespace = false;

        std::iter::from_fn(move || loop {
            if state.rest.is_empty() || state.failed {
                return None;
            }

            let mut old_loc = state.loc();
            match state.stack.last() {
                Some(LexerState::Template) => {
                    match state.rest.get(..2) {
                        Some("{{") => {
                            if state.rest.as_bytes().get(2) == Some(&b'-') {
                                state.advance(3);
                            } else {
                                state.advance(2);
                            }
                            state.stack.push(LexerState::InVariable);
                            return Some(Ok((Token::VariableStart, state.span(old_loc))));
                        }
                        Some("{%") => {
                            // raw blocks require some special handling.  If we are at the beginning of a raw
                            // block we want to skip everything until {% endraw %} completely ignoring iterior
                            // syntax and emit the entire raw block as TemplateData.
                            if let Some((mut ptr, _)) = skip_basic_tag(&state.rest[2..], "raw") {
                                ptr += 2;
                                while let Some(block) = memstr(&state.rest.as_bytes()[ptr..], b"{%")
                                {
                                    ptr += block + 2;
                                    if let Some((endraw, trim)) =
                                        skip_basic_tag(&state.rest[ptr..], "endraw")
                                    {
                                        let result = &state.rest[..ptr + endraw];
                                        state.advance(ptr + endraw);
                                        trim_leading_whitespace = trim;
                                        return Some(Ok((
                                            Token::TemplateData(result),
                                            state.span(old_loc),
                                        )));
                                    }
                                }
                                return Some(
                                    Err(state.syntax_error("unexpected end of raw block")),
                                );
                            }

                            if state.rest.as_bytes().get(2) == Some(&b'-') {
                                state.advance(3);
                            } else {
                                state.advance(2);
                            }

                            state.stack.push(LexerState::InBlock);
                            return Some(Ok((Token::BlockStart, state.span(old_loc))));
                        }
                        Some("{#") => {
                            if let Some(comment_end) = memstr(state.rest.as_bytes(), b"#}") {
                                if state
                                    .rest
                                    .as_bytes()
                                    .get(comment_end.saturating_sub(1))
                                    .copied()
                                    == Some(b'-')
                                {
                                    trim_leading_whitespace = true;
                                }
                                state.advance(comment_end + 2);
                                continue;
                            } else {
                                return Some(Err(state.syntax_error("unexpected end of comment")));
                            }
                        }
                        _ => {}
                    }

                    if trim_leading_whitespace {
                        trim_leading_whitespace = false;
                        state.skip_whitespace();
                        old_loc = state.loc();
                    }

                    let (lead, span) = match find_marker(state.rest) {
                        Some((start, false)) => (state.advance(start), state.span(old_loc)),
                        Some((start, _)) => {
                            let peeked = &state.rest[..start];
                            let trimmed = peeked.trim_end();
                            let lead = state.advance(trimmed.len());
                            let span = state.span(old_loc);
                            state.advance(peeked.len() - trimmed.len());
                            (lead, span)
                        }
                        None => (state.advance(state.rest.len()), state.span(old_loc)),
                    };
                    if lead.is_empty() {
                        continue;
                    }
                    return Some(Ok((Token::TemplateData(lead), span)));
                }
                Some(LexerState::InBlock | LexerState::InVariable) => {
                    // in blocks whitespace is generally ignored, skip it.
                    match state
                        .rest
                        .as_bytes()
                        .iter()
                        .position(|&x| !x.is_ascii_whitespace())
                    {
                        Some(0) => {}
                        None => {
                            state.advance(state.rest.len());
                            continue;
                        }
                        Some(offset) => {
                            state.advance(offset);
                            continue;
                        }
                    }

                    // look out for the end of blocks
                    if let Some(&LexerState::InBlock) = state.stack.last() {
                        if let Some("-%}") = state.rest.get(..3) {
                            state.stack.pop();
                            trim_leading_whitespace = true;
                            state.advance(3);
                            return Some(Ok((Token::BlockEnd, state.span(old_loc))));
                        }
                        if let Some("%}") = state.rest.get(..2) {
                            state.stack.pop();
                            state.advance(2);
                            return Some(Ok((Token::BlockEnd, state.span(old_loc))));
                        }
                    } else {
                        if let Some("-}}") = state.rest.get(..3) {
                            state.stack.pop();
                            state.advance(3);
                            trim_leading_whitespace = true;
                            return Some(Ok((Token::VariableEnd, state.span(old_loc))));
                        }
                        if let Some("}}") = state.rest.get(..2) {
                            state.stack.pop();
                            state.advance(2);
                            return Some(Ok((Token::VariableEnd, state.span(old_loc))));
                        }
                    }

                    // two character operators
                    let op = match state.rest.as_bytes().get(..2) {
                        Some(b"//") => Some(Token::FloorDiv),
                        Some(b"**") => Some(Token::Pow),
                        Some(b"==") => Some(Token::Eq),
                        Some(b"!=") => Some(Token::Ne),
                        Some(b">=") => Some(Token::Gte),
                        Some(b"<=") => Some(Token::Lte),
                        _ => None,
                    };
                    if let Some(op) = op {
                        state.advance(2);
                        return Some(Ok((op, state.span(old_loc))));
                    }

                    // single character operators (and strings)
                    let op = match state.rest.as_bytes().get(0) {
                        Some(b'+') => Some(Token::Plus),
                        Some(b'-') => Some(Token::Minus),
                        Some(b'*') => Some(Token::Mul),
                        Some(b'/') => Some(Token::Div),
                        Some(b'%') => Some(Token::Mod),
                        Some(b'!') => Some(Token::Bang),
                        Some(b'.') => Some(Token::Dot),
                        Some(b',') => Some(Token::Comma),
                        Some(b':') => Some(Token::Colon),
                        Some(b'~') => Some(Token::Tilde),
                        Some(b'|') => Some(Token::Pipe),
                        Some(b'=') => Some(Token::Assign),
                        Some(b'>') => Some(Token::Gt),
                        Some(b'<') => Some(Token::Lt),
                        Some(b'(') => Some(Token::ParenOpen),
                        Some(b')') => Some(Token::ParenClose),
                        Some(b'[') => Some(Token::BracketOpen),
                        Some(b']') => Some(Token::BracketClose),
                        Some(b'{') => Some(Token::BraceOpen),
                        Some(b'}') => Some(Token::BraceClose),
                        Some(b'\'') => {
                            return Some(state.eat_string(b'\''));
                        }
                        Some(b'"') => {
                            return Some(state.eat_string(b'"'));
                        }
                        Some(c) if c.is_ascii_digit() => return Some(state.eat_number()),
                        _ => None,
                    };
                    if let Some(op) = op {
                        state.advance(1);
                        return Some(Ok((op, state.span(old_loc))));
                    }

                    return Some(state.eat_identifier());
                }
                None => panic!("empty lexer state"),
            }
        })
    }

    mod error {
        use std::borrow::Cow;
        use std::fmt;

        use super::Span;

        /// Represents template errors.
        ///
        /// If debug mode is enabled a template error contains additional debug
        /// information that can be displayed by formatting an error with the
        /// alternative formatting (``format!("{:#}", err)``).  That information
        /// is also shown for the [`Debug`] display where the extended information
        /// is hidden when the alternative formatting is used.
        ///
        /// Since MiniJinja takes advantage of chained errors it's recommended
        /// to render the entire chain to better understand the causes.
        ///
        /// # Example
        ///
        /// Here is an example of you might want to render errors:
        ///
        /// ```rust
        /// # let mut env = minijinja::Environment::new();
        /// # env.add_template("", "");
        /// # let template = env.get_template("").unwrap(); let ctx = ();
        /// match template.render(ctx) {
        ///     Ok(result) => println!("{}", result),
        ///     Err(err) => {
        ///         eprintln!("Could not render template: {:#}", err);
        ///         // render causes as well
        ///         let mut err = &err as &dyn std::error::Error;
        ///         while let Some(next_err) = err.source() {
        ///             eprintln!();
        ///             eprintln!("caused by: {:#}", next_err);
        ///             err = next_err;
        ///         }
        ///     }
        /// }
        /// ```
        pub struct Error {
            repr: Box<ErrorRepr>,
        }

        /// The internal error data
        struct ErrorRepr {
            kind: ErrorKind,
            detail: Option<Cow<'static, str>>,
            name: Option<String>,
            lineno: usize,
            span: Option<Span>,
            source: Option<Box<dyn std::error::Error + Send + Sync>>,
            #[cfg(feature = "debug")]
            debug_info: Option<crate::debug::DebugInfo>,
        }

        impl fmt::Debug for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut err = f.debug_struct("Error");
                err.field("kind", &self.kind());
                if let Some(ref detail) = self.repr.detail {
                    err.field("detail", detail);
                }
                if let Some(ref name) = self.name() {
                    err.field("name", name);
                }
                if let Some(line) = self.line() {
                    err.field("line", &line);
                }
                if let Some(ref source) = std::error::Error::source(self) {
                    err.field("source", source);
                }
                err.finish()?;

                // so this is a bit questionablem, but because of how commonly errors are just
                // unwrapped i think it's sensible to spit out the debug info following the
                // error struct dump.
                #[cfg(feature = "debug")]
                {
                    if !f.alternate() {
                        if let Some(info) = self.debug_info() {
                            Ok(writeln!(f));
                            Ok(crate::debug::render_debug_info(
                                f,
                                self.name(),
                                self.kind(),
                                self.line(),
                                self.span(),
                                info,
                            ));
                            Ok(writeln!(f));
                        }
                    }
                }

                Ok(())
            }
        }

        /// An enum describing the error kind.
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub enum ErrorKind {
            /// A non primitive value was encountered where one was expected.
            NonPrimitive,
            /// A value is not valid for a key in a map.
            NonKey,
            /// An invalid operation was attempted.
            InvalidOperation,
            /// The template has a syntax error
            SyntaxError,
            /// A template was not found.
            TemplateNotFound,
            /// Too many arguments were passed to a function.
            TooManyArguments,
            /// A expected argument was missing
            MissingArgument,
            /// A filter is unknown
            UnknownFilter,
            /// A test is unknown
            UnknownTest,
            /// A function is unknown
            UnknownFunction,
            /// Un unknown method was called
            UnknownMethod,
            /// A bad escape sequence in a string was encountered.
            BadEscape,
            /// An operation on an undefined value was attempted.
            UndefinedError,
            /// Not able to serialize this value.
            BadSerialization,
            /// An error happened in an include.
            BadInclude,
            /// An error happened in a super block.
            EvalBlock,
            /// Unable to unpack a value.
            CannotUnpack,
            /// Failed writing output.
            WriteFailure,
            /// Engine ran out of fuel
            #[cfg(feature = "fuel")]
            OutOfFuel,
        }

        impl ErrorKind {
            fn description(self) -> &'static str {
                match self {
                    ErrorKind::NonPrimitive => "not a primitive",
                    ErrorKind::NonKey => "not a key type",
                    ErrorKind::InvalidOperation => "invalid operation",
                    ErrorKind::SyntaxError => "syntax error",
                    ErrorKind::TemplateNotFound => "template not found",
                    ErrorKind::TooManyArguments => "too many arguments",
                    ErrorKind::MissingArgument => "missing argument",
                    ErrorKind::UnknownFilter => "unknown filter",
                    ErrorKind::UnknownFunction => "unknown function",
                    ErrorKind::UnknownTest => "unknown test",
                    ErrorKind::UnknownMethod => "unknown method",
                    ErrorKind::BadEscape => "bad string escape",
                    ErrorKind::UndefinedError => "undefined value",
                    ErrorKind::BadSerialization => "could not serialize to internal format",
                    ErrorKind::BadInclude => "could not render include",
                    ErrorKind::EvalBlock => "could not render block",
                    ErrorKind::CannotUnpack => "cannot unpack",
                    ErrorKind::WriteFailure => "failed to write output",
                    #[cfg(feature = "fuel")]
                    ErrorKind::OutOfFuel => "engine ran out of fuel",
                }
            }
        }

        impl fmt::Display for ErrorKind {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.description())
            }
        }

        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if let Some(ref detail) = self.repr.detail {
                    write!(f, "{}: {}", self.kind(), detail);
                } else {
                    write!(f, "{}", self.kind());
                }
                if let Some(ref filename) = self.name() {
                    write!(f, " (in {}:{})", filename, self.line().unwrap_or(0));
                }
                #[cfg(feature = "debug")]
                {
                    if f.alternate() {
                        if let Some(info) = self.debug_info() {
                            Ok(crate::debug::render_debug_info(
                                f,
                                self.name(),
                                self.kind(),
                                self.line(),
                                self.span(),
                                info,
                            ));
                        }
                    }
                }
                Ok(())
            }
        }

        impl Error {
            /// Creates a new error with kind and detail.
            pub fn new<D: Into<Cow<'static, str>>>(kind: ErrorKind, detail: D) -> Error {
                Error {
                    repr: Box::new(ErrorRepr {
                        kind,
                        detail: Some(detail.into()),
                        name: None,
                        lineno: 0,
                        span: None,
                        source: None,
                        #[cfg(feature = "debug")]
                        debug_info: None,
                    }),
                }
            }

            pub(crate) fn set_filename_and_line(&mut self, filename: &str, lineno: usize) {
                self.repr.name = Some(filename.into());
                self.repr.lineno = lineno;
            }

            pub(crate) fn set_filename_and_span(&mut self, filename: &str, span: Span) {
                self.repr.name = Some(filename.into());
                self.repr.span = Some(span);
                self.repr.lineno = span.start_line;
            }

            pub(crate) fn new_not_found(name: &str) -> Error {
                Error::new(
                    ErrorKind::TemplateNotFound,
                    format!("template {name:?} does not exist"),
                )
            }

            /// Attaches another error as source to this error.
            #[allow(unused)]
            pub fn with_source<E: std::error::Error + Send + Sync + 'static>(
                mut self,
                source: E,
            ) -> Self {
                self.repr.source = Some(Box::new(source));
                self
            }

            /// Returns the error kind
            pub fn kind(&self) -> ErrorKind {
                self.repr.kind
            }

            /// Returns the filename of the template that caused the error.
            pub fn name(&self) -> Option<&str> {
                self.repr.name.as_deref()
            }

            /// Returns the line number where the error occurred.
            pub fn line(&self) -> Option<usize> {
                if self.repr.lineno > 0 {
                    Some(self.repr.lineno)
                } else {
                    None
                }
            }

            /// Returns the line number where the error occurred.
            #[cfg(feature = "debug")]
            pub(crate) fn span(&self) -> Option<Span> {
                self.repr.span
            }

            /// Returns the template debug information is available.
            ///
            /// The debug info snapshot is only embedded into the error if the debug
            /// mode is enabled on the environment
            /// ([`Environment::set_debug`](crate::Environment::set_debug)).
            #[cfg(feature = "debug")]
            #[cfg_attr(docsrs, doc(cfg(feature = "debug")))]
            pub(crate) fn debug_info(&self) -> Option<&crate::debug::DebugInfo> {
                self.repr.debug_info.as_ref()
            }

            #[cfg(feature = "debug")]
            #[cfg_attr(docsrs, doc(cfg(feature = "debug")))]
            pub(crate) fn attach_debug_info(&mut self, value: crate::debug::DebugInfo) {
                self.repr.debug_info = Some(value);
            }
        }

        impl std::error::Error for Error {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.repr.source.as_ref().map(|err| err.as_ref() as _)
            }
        }

        impl From<ErrorKind> for Error {
            fn from(kind: ErrorKind) -> Self {
                Error {
                    repr: Box::new(ErrorRepr {
                        kind,
                        detail: None,
                        name: None,
                        lineno: 0,
                        span: None,
                        source: None,
                        #[cfg(feature = "debug")]
                        debug_info: None,
                    }),
                }
            }
        }

        impl From<fmt::Error> for Error {
            fn from(_: fmt::Error) -> Self {
                Error::new(ErrorKind::WriteFailure, "formatting failed")
            }
        }

        impl serde::ser::Error for Error {
            fn custom<T>(msg: T) -> Self
            where
                T: fmt::Display,
            {
                Error::new(ErrorKind::BadSerialization, msg.to_string())
            }
        }

        pub fn attach_basic_debug_info<T>(rv: Result<T, Error>, source: &str) -> Result<T, Error> {
            #[cfg(feature = "debug")]
            {
                match rv {
                    Ok(rv) => Ok(rv),
                    Err(mut err) => {
                        err.repr.debug_info = Some(crate::debug::DebugInfo {
                            template_source: Some(source.to_string()),
                            ..Default::default()
                        });
                        Err(err)
                    }
                }
            }
            #[cfg(not(feature = "debug"))]
            {
                let _source = source;
                rv
            }
        }
    }

    mod utils {
        use std::char::decode_utf16;
        use std::collections::BTreeMap;
        use std::fmt;
        use std::iter::{once, repeat};
        use std::str::Chars;

        use super::error::{Error, ErrorKind};

        #[cfg(test)]
        use similar_asserts::assert_eq;

        /// internal marker to seal up some trait methods
        pub struct SealedMarker;

        pub fn memchr(haystack: &[u8], needle: u8) -> Option<usize> {
            haystack.iter().position(|&x| x == needle)
        }

        pub fn memstr(haystack: &[u8], needle: &[u8]) -> Option<usize> {
            haystack
                .windows(needle.len())
                .position(|window| window == needle)
        }

        fn invalid_autoescape(name: &str) -> Result<(), Error> {
            Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("Default formatter does not know how to format to custom format '{name}'"),
            ))
        }

        /// Controls the autoescaping behavior.
        ///
        /// For more information see
        /// [`set_auto_escape_callback`](crate::Environment::set_auto_escape_callback).
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum AutoEscape {
            /// Do not apply auto escaping.
            None,
            /// Use HTML auto escaping rules.
            ///
            /// Any value will be converted into a string and the following characters
            /// will be escaped in ways compatible to XML and HTML: `<`, `>`, `&`, `"`,
            /// `'`, and `/`.
            Html,
            /// Use escaping rules suitable for JSON/JavaScript or YAML.
            ///
            /// Any value effectively ends up being serialized to JSON upon printing.  The
            /// serialized values will be compatible with JavaScript and YAML as well.
            #[cfg(feature = "json")]
            #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
            Json,
            /// A custom auto escape format.
            ///
            /// The default formatter does not know how to deal with a custom escaping
            /// format and would error.  The use of these requires a custom formatter.
            /// See [`set_formatter`](crate::Environment::set_formatter).
            Custom(&'static str),
        }

        /// Helper to HTML escape a string.
        pub struct HtmlEscape<'a>(pub &'a str);

        impl<'a> fmt::Display for HtmlEscape<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(feature = "v_htmlescape")]
                {
                    fmt::Display::fmt(&v_htmlescape::escape(self.0), f)
                }
                // this is taken from askama-escape
                #[cfg(not(feature = "v_htmlescape"))]
                {
                    let bytes = self.0.as_bytes();
                    let mut start = 0;

                    for (i, b) in bytes.iter().enumerate() {
                        macro_rules! escaping_body {
                            ($quote:expr) => {{
                                if start < i {
                                    f.write_str(unsafe {
                                        std::str::from_utf8_unchecked(&bytes[start..i])
                                    })?;
                                }
                                f.write_str($quote)?;
                                start = i + 1;
                            }};
                        }
                        if b.wrapping_sub(b'"') <= b'>' - b'"' {
                            match *b {
                                b'<' => escaping_body!("&lt;"),
                                b'>' => escaping_body!("&gt;"),
                                b'&' => escaping_body!("&amp;"),
                                b'"' => escaping_body!("&quot;"),
                                b'\'' => escaping_body!("&#x27;"),
                                b'/' => escaping_body!("&#x2f;"),
                                _ => (),
                            }
                        }
                    }

                    if start < bytes.len() {
                        f.write_str(unsafe { std::str::from_utf8_unchecked(&bytes[start..]) })
                    } else {
                        Ok(())
                    }
                }
            }
        }

        struct Unescaper {
            out: String,
            pending_surrogate: u16,
        }

        impl Unescaper {
            fn unescape(mut self, s: &str) -> Result<String, Error> {
                let mut char_iter = s.chars();

                while let Some(c) = char_iter.next() {
                    if c == '\\' {
                        match char_iter.next() {
                            None => return Err(ErrorKind::BadEscape.into()),
                            Some(d) => match d {
                                '"' | '\\' | '/' | '\'' => {
                                    self.push_char(d)?;
                                }
                                'b' => {
                                    self.push_char('\x08')?;
                                }
                                'f' => {
                                    self.push_char('\x0C')?;
                                }
                                'n' => {
                                    self.push_char('\n')?;
                                }
                                'r' => {
                                    self.push_char('\r')?;
                                }
                                't' => {
                                    self.push_char('\t')?;
                                }
                                'u' => {
                                    let val = self.parse_u16(&mut char_iter)?;
                                    self.push_u16(val);
                                }
                                _ => return Err(ErrorKind::BadEscape.into()),
                            },
                        }
                    } else {
                        self.push_char(c)?;
                    }
                }

                if self.pending_surrogate != 0 {
                    Err(ErrorKind::BadEscape.into())
                } else {
                    Ok(self.out)
                }
            }

            fn parse_u16(&self, chars: &mut Chars) -> Result<u16, Error> {
                let hexnum = chars.chain(repeat('\0')).take(4).collect::<String>();
                u16::from_str_radix(&hexnum, 16).map_err(|_| ErrorKind::BadEscape.into())
            }

            fn push_u16(&mut self, c: u16) -> Result<(), Error> {
                match (self.pending_surrogate, (0xD800..=0xDFFF).contains(&c)) {
                    (0, false) => match decode_utf16(once(c)).next() {
                        Some(Ok(c)) => self.out.push(c),
                        _ => return Err(ErrorKind::BadEscape.into()),
                    },
                    (_, false) => return Err(ErrorKind::BadEscape.into()),
                    (0, true) => self.pending_surrogate = c,
                    (prev, true) => match decode_utf16(once(prev).chain(once(c))).next() {
                        Some(Ok(c)) => {
                            self.out.push(c);
                            self.pending_surrogate = 0;
                        }
                        _ => return Err(ErrorKind::BadEscape.into()),
                    },
                }
                Ok(())
            }

            fn push_char(&mut self, c: char) -> Result<(), Error> {
                if self.pending_surrogate != 0 {
                    Err(ErrorKind::BadEscape.into())
                } else {
                    self.out.push(c);
                    Ok(())
                }
            }
        }

        /// Un-escape a string, following JSON rules.
        pub fn unescape(s: &str) -> Result<String, Error> {
            Unescaper {
                out: String::new(),
                pending_surrogate: 0,
            }
            .unescape(s)
        }

        pub struct BTreeMapKeysDebug<'a, K: fmt::Debug, V>(pub &'a BTreeMap<K, V>);

        impl<'a, K: fmt::Debug, V> fmt::Debug for BTreeMapKeysDebug<'a, K, V> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.0.iter().map(|x| x.0)).finish()
            }
        }

        pub struct OnDrop<F: FnOnce()>(Option<F>);

        impl<F: FnOnce()> OnDrop<F> {
            pub fn new(f: F) -> Self {
                Self(Some(f))
            }
        }

        impl<F: FnOnce()> Drop for OnDrop<F> {
            fn drop(&mut self) {
                self.0.take().unwrap()();
            }
        }

        #[test]
        fn test_html_escape() {
            let input = "<>&\"'/";
            let output = HtmlEscape(input).to_string();
            assert_eq!(output, "&lt;&gt;&amp;&quot;&#x27;&#x2f;");
        }

        #[test]
        fn test_unescape() {
            assert_eq!(unescape(r"foo\u2603bar").unwrap(), "foo\u{2603}bar");
            assert_eq!(unescape(r"\t\b\f\r\n\\\/").unwrap(), "\t\x08\x0c\r\n\\/");
            assert_eq!(unescape("foobarbaz").unwrap(), "foobarbaz");
            assert_eq!(unescape(r"\ud83d\udca9").unwrap(), "💩");
        }
    }

    #[cfg(test)]
    mod test {
        use similar_asserts::assert_eq;

        use super::*;

        #[test]
        fn test_find_marker() {
            assert!(find_marker("{").is_none());
            assert!(find_marker("foo").is_none());
            assert!(find_marker("foo {").is_none());
            assert_eq!(find_marker("foo {{"), Some((4, false)));
            assert_eq!(find_marker("foo {{-"), Some((4, true)));
        }

        #[test]
        fn test_is_basic_tag() {
            assert_eq!(skip_basic_tag(" raw %}", "raw"), Some((7, false)));
            assert_eq!(skip_basic_tag(" raw %}", "endraw"), None);
            assert_eq!(skip_basic_tag("  raw  %}", "raw"), Some((9, false)));
            assert_eq!(skip_basic_tag("-  raw  -%}", "raw"), Some((11, true)));
        }

        #[test]
        fn test_basic_identifiers() {
            fn assert_ident(s: &str) {
                match tokenize(s, true).next() {
                    Some(Ok((Token::Ident(ident), _))) if ident == s => {}
                    _ => panic!("did not get a matching token result: {s:?}"),
                }
            }

            fn assert_not_ident(s: &str) {
                let res = tokenize(s, true).collect::<Result<Vec<_>, _>>();
                if let Ok(tokens) = res {
                    if let &[(Token::Ident(_), _)] = &tokens[..] {
                        panic!("got a single ident for {s:?}")
                    }
                }
            }

            assert_ident("foo_bar_baz");
            assert_ident("_foo_bar_baz");
            assert_ident("_42world");
            assert_ident("_world42");
            assert_ident("world42");
            assert_not_ident("42world");

            #[cfg(feature = "unicode")]
            {
                assert_ident("foo");
                assert_ident("föö");
                assert_ident("き");
                assert_ident("_");
                assert_not_ident("1a");
                assert_not_ident("a-");
                assert_not_ident("🐍a");
                assert_not_ident("a🐍🐍");
                assert_ident("ᢅ");
                assert_ident("ᢆ");
                assert_ident("℘");
                assert_ident("℮");
                assert_not_ident("·");
                assert_ident("a·");
            }
        }
    }
}
