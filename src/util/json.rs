//! JSON implementation

use std::collections::HashMap;

/// JSON object structure
#[derive(Clone, PartialEq)]
pub enum Json {
    /// String (unescaped)
    String(String),

    /// Object (set of key-value pairs)
    Object(HashMap<String, Json>),

    /// Array (array)
    Array(Vec<Json>),

    /// Boolean (true/false)
    Boolean(bool),

    /// Some numeric constant
    Number(f64),

    /// Null value (null)
    Null,
}

/// Type of the JSON item
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum JsonType {
    String,
    Object,
    Array,
    Boolean,
    Number,
    Null,
}

/// Wrong json item type
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct JsonItemTypeError {
    /// Actual json item type
    pub actual: JsonType,

    /// Required json item type
    pub required: JsonType,
}

/// .JSON text paring error
#[derive(Debug)]
pub enum JsonParsingError {
    /// Unknown character in escape sequence
    UnknownEscapeSequence(char),

    /// Input ended but string is not closed
    UnexpectedStringEnd,

    /// Missing array trailing symbol
    ArrayEndSymbolMissing,

    /// Missing array trailing symbol
    ObjectEndSymbolMissing,

    /// Colon of the object item missing
    ObjectItemColonMissing,

    /// Text is not numeric constant
    ParseFloatError(std::num::ParseFloatError),

    /// Input string is empty
    EmptyString,
}

impl Json {
    fn parse_string(source: &str) -> Result<(String, &str), JsonParsingError> {
        let mut len = 0;
        let mut chars = source.chars().map(|ch| { len += ch.len_utf8(); ch });

        // First character must exist due to call contract
        chars.next().unwrap();

        let mut result = String::new();
        let mut is_escape = false;

        while let Some(ch) = chars.next() {
            if is_escape {
                let escaped = match ch {
                    '\"' => '\"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{0008}',
                    'f' => '\u{000C}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(JsonParsingError::UnknownEscapeSequence(ch)),
                };
                result.push(escaped);
                is_escape = false;
                continue;
            }

            if ch == '\\' {
                is_escape = true;
                continue;
            }

            if ch == '\"' {
                return Ok((result, &source[len..]));
            }

            result.push(ch);
        }

        Err(JsonParsingError::UnexpectedStringEnd)
    }

    fn parse_array(mut source: &str) -> Result<(Vec<Self>, &str), JsonParsingError> {
        source = &source['['.len_utf8()..].trim_start();
        if source.starts_with(']') {
            return Ok((Vec::new(), &source[']'.len_utf8()..]))
        }

        let mut result = Vec::new();

        loop {
            source = source.trim_start();
            let item;
            (item, source) = Self::parse_json(source)?;
            result.push(item);

            source = source.trim_start();
            if source.starts_with(',') {
                source = &source[','.len_utf8()..];
            } else {
                // Trim whitespace
                source = source.trim_start();

                return if source.starts_with(']') {
                    Ok((result, &source[']'.len_utf8()..]))
                } else {
                    Err(JsonParsingError::ArrayEndSymbolMissing)
                };
            }
        }
    }

    fn parse_object(mut source: &str) -> Result<(HashMap<String, Self>, &str), JsonParsingError> {
        source = &source['{'.len_utf8()..].trim_start();
        if source.starts_with('}') {
            return Ok((HashMap::new(), &source['}'.len_utf8()..]))
        }

        let mut result = HashMap::new();

        loop {
            source = source.trim_start();
            let key;
            (key, source) = Self::parse_string(source)?;
            source = source.trim_start();
            if !source.starts_with(':') {
                return Err(JsonParsingError::ObjectItemColonMissing);
            }
            source = &source[':'.len_utf8()..];
            let value;
            (value, source) = Self::parse_json(source)?;

            result.insert(key, value);

            source = source.trim_start();
            if source.starts_with(',') {
                source = &source[','.len_utf8()..];
            } else {
                // Trim whitespace
                source = source.trim_start();

                return if source.starts_with('}') {
                    Ok((result, &source['}'.len_utf8()..]))
                } else {
                    Err(JsonParsingError::ObjectEndSymbolMissing)
                };
            }
        }
    }

    fn parse_json(source: &str) -> Result<(Self, &str), JsonParsingError> {
        let source = source.trim_start();
        let first_char = source.chars().nth(0).ok_or(JsonParsingError::EmptyString)?;

        match first_char {
            // String, Object, Array
            '\"' => Self::parse_string(source).map(|(str, rest)| (Json::String(str), rest)),
            '{' => Self::parse_object(source).map(|(obj, rest)| (Json::Object(obj), rest)),
            '[' => Self::parse_array(source).map(|(arr, rest)| (Json::Array(arr), rest)),

            // True, False, Null
            _ if source.starts_with("true") => Ok((Json::Boolean(true), &source["true".len()..])),
            _ if source.starts_with("false") => Ok((Json::Boolean(false), &source["false".len()..])),
            _ if source.starts_with("null") => Ok((Json::Null, &source["null".len()..])),

            // Number
            _ => {
                // No panic because 'source' string is not empty
                let n_word = source
                    .split_terminator(|ch: char| ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == '-')
                    .next()
                    .unwrap();

                let n = n_word.parse::<f64>().map_err(JsonParsingError::ParseFloatError)?;

                Ok((Json::Number(n), &source[n_word.len()..]))
            }
        }
    }

    /// Parse JSON file
    pub fn parse(source: &str) -> Result<Self, JsonParsingError> {
        Self::parse_json(source).map(|v| v.0)
    }

    /// Display json to some output buffer
    ///
    /// # Note
    /// Bytes written to the `out` are contents of a UTF-8 string.
    pub fn display(&self, out: &mut dyn std::io::Write, tab_size: usize) {
        /// JSON formatter
        struct JsonFormatter<'t> {
            /// Size of the tab
            tab_size: usize,

            /// Output buffer
            out: &'t mut dyn std::io::Write,
        }

        impl<'t> JsonFormatter<'t> {
            fn write_indent(&mut self, amount: usize) -> Result<(), std::io::Error> {
                write!(self.out, "{: <1$}}}", "", self.tab_size * amount)
            }

            fn write_str(&mut self, str: &str) -> Result<(), std::io::Error> {
                self.out.write(str.as_bytes()).map(|_| ())
            }

            fn write_string(&mut self, str: &str) -> Result<(), std::io::Error> {
                self.write_str("\"")?;
                for ch in str.chars() {
                    write!(self.out, "\\u{{{:X}}}", ch as u32)?;
                }
                self.write_str("\"")?;

                Ok(())
            }

            fn write_object_kv(&mut self, kv: (&String, &Json), depth: usize) -> Result<(), std::io::Error> {
                self.write_string(kv.0)?;
                self.write_str(": ")?;
                self.write_json(kv.1, depth + 1)?;

                Ok(())
            }

            /// Write some iterated object
            fn write_iterated<T>(
                &mut self,
                depth: usize,
                first: char,
                last: char,
                mut iter: impl Iterator<Item = T>,
                mut write_item: impl FnMut(&mut Self, T, usize) -> Result<(), std::io::Error>
            ) -> Result<(), std::io::Error> {
                let Some(i0) = iter.next() else {
                    write!(self.out, "{}{}", first, last)?;
                    return Ok(());
                };
                write!(self.out, "{}", first)?;
                self.write_indent(depth + 1)?;
                write_item(self, i0, depth)?;

                for i in iter {
                    self.write_str(",\n")?;
                    self.write_indent(depth + 1)?;
                    write_item(self, i, depth)?;
                }

                self.write_str("\n")?;
                self.write_indent(depth)?;
                write!(self.out, "{}", last)?;

                Ok(())
            }

            fn write_json(&mut self, json: &Json, depth: usize) -> Result<(), std::io::Error> {
                match json {
                    Json::String(str) => self.write_string(str.as_str())?,
                    Json::Object(obj) => self.write_iterated(depth, '{', '}', obj.iter(), Self::write_object_kv)?,
                    Json::Array(arr) => self.write_iterated(depth, '[', ']', arr.iter(), Self::write_json)?,
                    Json::Boolean(bool) => self.write_str(if *bool { "true" } else { "false" })?,
                    Json::Number(num) => write!(self.out, "{}", num)?,
                    Json::Null => self.write_str("null")?,
                }

                Ok(())
            }

            pub fn write(&mut self, json: &Json) -> Result<(), std::io::Error> {
                self.write_json(json, 0)
            }
        }

        // Perform actual write
        _ = JsonFormatter { out, tab_size }.write(self);
    }

    /// Convert json into formatted string
    pub fn to_string(&self) -> String {
        let mut data = Vec::new();
        self.display(&mut data, 4);
        unsafe { String::from_utf8_unchecked(data) }
    }

    /// Get self type
    pub const fn ty(&self) -> JsonType {
        match self {
            Self::String(_) => JsonType::String,
            Self::Object(_) => JsonType::Object,
            Self::Array(_) => JsonType::Array,
            Self::Boolean(_) => JsonType::Boolean,
            Self::Number(_) => JsonType::Number,
            Self::Null => JsonType::Null,
        }
    }

    /// Generate type error
    fn type_error<T>(&self, required: JsonType) -> Result<T, JsonItemTypeError> {
        Err(JsonItemTypeError { actual: self.ty(), required })
    }

    /// Try to interpret self as a string
    pub fn as_string(&self) -> Result<&str, JsonItemTypeError> {
        match self {
            Self::String(str) => Ok(str.as_ref()),
            _ => self.type_error(JsonType::String),
        }
    }

    /// Try to interpret self as object
    pub fn as_object(&self) -> Result<&HashMap<String, Self>, JsonItemTypeError> {
        match self {
            Self::Object(obj) => Ok(&obj),
            _ => self.type_error(JsonType::Object),
        }
    }

    /// Try to interpret self as array
    pub fn as_array(&self) -> Result<&[Json], JsonItemTypeError> {
        match self {
            Self::Array(arr) => Ok(arr.as_slice()),
            _ => self.type_error(JsonType::Array),
        }
    }

    /// Try to interpret self as number
    pub fn as_number(&self) -> Result<f64, JsonItemTypeError> {
        match self {
            Self::Number(num) => Ok(*num),
            _ => self.type_error(JsonType::Number),
        }
    }

    /// Try to interpret self as boolean
    pub fn as_boolean(&self) -> Result<bool, JsonItemTypeError> {
        match self {
            Self::Boolean(bool) => Ok(*bool),
            _ => self.type_error(JsonType::Boolean),
        }
    }

    /// Try to interpret self as null
    pub fn as_null(&self) -> Result<(), JsonItemTypeError> {
        match self {
            Self::Null => Ok(()),
            _ => self.type_error(JsonType::Null),
        }
    }
}
