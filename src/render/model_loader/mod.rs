//! Model loader implementation file

use std::collections::HashMap;
use crate::{math, render::core::Vertex};

/// Face component kind
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FaceComponent {
    /// Vertex component
    Vertex,

    /// Texture coordinate component
    TexCoord,

    /// Normal component
    Normal,
}

/// Error of parsing .obj file
#[derive(Debug)]
pub enum ParsingError {

    /// Vector (vertex/texcoord/normal) components missing
    VectorComponentsMissing {
        /// Line error found at
        line_number: u32,

        /// Expected amount of vector components
        expected: u32,

        /// Actual amount
        actual: u32,
    },

    /// Float-point number parsing error
    FloatParsingError {
        /// Line number
        line_number: u32,

        /// Float-point number parsing error
        error: std::num::ParseFloatError,
    },

    /// Integer number parsing error
    IntParsingError {
        /// Line number
        line_number: u32,

        /// Error itself
        error: std::num::ParseIntError,
    },

    /// Bad index of the face component
    BadFaceIndex {
        /// Kind of face component
        component: FaceComponent,

        /// Current length of the component pool
        pool_len: u32,

        /// Index
        index: u32,
    },

    /// Face triple does not contain any numbers
    EmptyFaceTriple {
        line_number: u32,
    }
}

/// Parse face number
fn parse_face_num(line_number: u32, str: &str) -> Result<u32, ParsingError> {
    if str.is_empty() {
        Ok(0)
    } else {
        str.parse::<u32>().map_err(|error| ParsingError::IntParsingError { line_number, error })
    }
}

/// Parse face triple (e.g. x/y/z or x/y or x into triple)
fn parse_face(line_number: u32, str: &str) -> Result<(u32, u32, u32), ParsingError> {
    let mut iter = str.split('/');

    let pi = match iter.next() {
        Some(s) => parse_face_num(line_number, s)?,
        None => return Err(ParsingError::EmptyFaceTriple { line_number }),
    };
    let ti = match iter.next() {
        Some(s) => parse_face_num(line_number, s)?,
        None => return Ok((pi, 0, 0)),
    };
    let ni = match iter.next() {
        Some(s) => parse_face_num(line_number, s)?,
        None => return Ok((pi, ti, 0)),
    };

    Ok((pi, ti, ni))
}

/// .OBJ file mesh
pub struct ObjMesh {
    /// Vertex set
    pub vertices: Vec<Vertex>,

    /// Index set
    pub indices: Vec<u32>,
}

/// Parse .obj file
pub fn parse(str: &str) -> Result<ObjMesh, ParsingError> {
    let mut positions = vec![math::vector![0.0f32, 0.0, 0.0]];
    let mut normals = vec![0u32];
    let mut tex_coords = vec![math::vector![0.0f32, 0.0]];

    let mut vertex_map = HashMap::<(u32, u32, u32), u32>::new();

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut word_vec = Vec::new();

    let iter  = str.lines().enumerate().map(|(index, line)| (line, (index as u32) + 1));

    for (line, line_number) in iter {
        word_vec.clear();
        word_vec.extend(line.split_whitespace().filter(|w| w.len() != 0));

        let Some(kind) = word_vec.get(0).copied() else {
            continue;
        };

        match kind {

            // Vertex (e.g. position)
            "v" => {
                let Some(&[x, y, z]) = word_vec.get(1..4) else {
                    return Err(ParsingError::VectorComponentsMissing {
                        line_number,
                        expected: 3,
                        actual: word_vec.len() as u32 - 1
                    });
                };

                let [x, y, z] = [x, y, z].map(str::parse::<f32>);

                let zip = x
                    .and_then(|x| y.map(|y| (x, y)))
                    .and_then(|(x, y)| z.map(|z| (x, y, z)));

                let (x, y, z) = match zip {
                    Ok(t) => t,
                    Err(error) => return Err(ParsingError::FloatParsingError { line_number, error }),
                };

                positions.push(math::vector![x, y, z]);
            }

            // Texture coordinates
            "vt" => {
                let Some(&[x, y]) = word_vec.get(1..3) else {
                    return Err(ParsingError::VectorComponentsMissing {
                        line_number,
                        expected: 2,
                        actual: word_vec.len() as u32 - 1
                    });
                };

                let [x, y] = [x, y].map(str::parse::<f32>);

                let zip = x
                    .and_then(|x| y.map(|y| (x, y)));

                let (x, y) = match zip {
                    Ok(t) => t,
                    Err(error) => return Err(ParsingError::FloatParsingError { line_number, error }),
                };

                tex_coords.push(math::vector![x, y]);
            }

            // Normal
            "vn" => {
                let Some(&[x, y, z]) = word_vec.get(1..4) else {
                    return Err(ParsingError::VectorComponentsMissing {
                        line_number,
                        expected: 3,
                        actual: word_vec.len() as u32 - 1
                    });
                };

                let [x, y, z] = [x, y, z].map(str::parse::<f32>);

                let zip = x
                    .and_then(|x| y.map(|y| (x, y)))
                    .and_then(|(x, y)| z.map(|z| (x, y, z)));

                let (x, y, z) = match zip {
                    Ok(t) => t,
                    Err(error) => return Err(ParsingError::FloatParsingError { line_number, error }),
                };

                // Normalized input normal and pack it in octmap
                let normalized = math::FVec::new3(x, y, z).normalized();
                let packed = Vertex::pack_direction_octmap(normalized.x(), normalized.y(), normalized.z());
                normals.push(packed);
            }

            // Face
            "f" => {

                // Parse word into face triple and get triple's index in the vertex array
                let mut vt_index = |word: &str| -> Result<u32, ParsingError> {
                    let tup = parse_face(line_number, word)?;

                    Ok(match vertex_map.entry(tup) {
                        std::collections::hash_map::Entry::Occupied(occ) => *occ.get(),
                        std::collections::hash_map::Entry::Vacant(vac) => {
                            let (pi, ti, ni) = tup;

                            let index = vertices.len() as u32;

                            vertices.push(Vertex {
                                position: positions
                                    .get(pi as usize)
                                    .copied()
                                    .ok_or_else(|| ParsingError::BadFaceIndex {
                                        component: FaceComponent::Vertex,
                                        pool_len: positions.len() as u32,
                                        index: pi,
                                    })?,
                                tex_coord: tex_coords
                                    .get(ti as usize)
                                    .copied()
                                    .ok_or_else(|| ParsingError::BadFaceIndex {
                                        component: FaceComponent::TexCoord,
                                        pool_len: tex_coords.len() as u32,
                                        index: ti,
                                    })?,
                                normal: normals
                                    .get(ni as usize)
                                    .copied()
                                    .ok_or_else(|| ParsingError::BadFaceIndex {
                                        component: FaceComponent::Normal,
                                        pool_len: normals.len() as u32,
                                        index: ni,
                                    })?,
                                tangent: 0,
                                misc: 0,
                            });

                            vac.insert(index);

                            index
                        }
                    })
                };

                let Some(&[wbase, wcurr]) = word_vec.get(1..3) else {
                    continue;
                };

                let ibase = vt_index(wbase)?;
                let mut icurr = vt_index(wcurr)?;

                // Do not account single-side face?
                for word in word_vec.iter().skip(3) {
                    let i = vt_index(word)?;

                    indices.push(ibase);
                    indices.push(icurr);
                    indices.push(i);

                    icurr = i;
                }
            }

            // Idk
            _ => continue,
        }
    }

    Ok(ObjMesh {
        vertices,
        indices
    })
}
