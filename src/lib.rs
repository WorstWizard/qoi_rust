//! # QOI Encoder/Decoder
//! This library implements a minimal QOI image format encoder and decoder. Read more at [https://qoiformat.org](https://qoiformat.org).
//! The interface uses row-major packed arrays of bytes, so buffers must have the layout:
//! ```
//! /*
//! [R, G, B (, A)], ..., width,
//! ...,
//! ...,
//! ...,
//! height
//! 
//! Total number of bytes: channels * width * height
//! */
//! ```
//! 
//! To encode an image, create a [`QOIHeader`] matching the input buffer and use [`encode()`].
//! To decode an image, use [`decode()`] on the input byte array.

/// The header defines the size, number of color channels, and color space of the image.
/// 
/// Create a new header like so:
/// ```
/// use qoi_format::*;
/// 
/// let header = QOIHeader::new()
///     .size(256, 256)
///     .channels(3) //Must be 3 or 4
///     .colorspace(LINEAR) //Constants for convenience, equivalent to below
///     .colorspace(0); //0 = SRGB, 1 = All linear. Does not affect encoding
/// ```
#[derive(Clone, Copy)]
pub struct QOIHeader {
    width: u32,
    height: u32,
    channels: u8, //3 = RGB, 4 = RGBA
    colorspace: u8, //0 = SRGB, 1 = all channels linear
}

const MAGIC_BYTES: [u8; 4] = *b"qoif";

/// Use as parameter to [`QOIHeader::channels`].
pub const SRGB: u8 = 0;
/// Use as parameter to [`QOIHeader::channels`].
pub const LINEAR: u8 = 1;

impl QOIHeader {
    /// Create a new header with no specified parameters. This *will* encode/decode, but a 0x0 pixel image likely won't play well with most readers.
    pub fn new() -> QOIHeader {
        QOIHeader {
            width: 0,
            height: 0,
            channels: 3,
            colorspace: 0,
        }
    }
    /// Set the width and height of the image.
    pub fn size(mut self, width: u32, height: u32) -> QOIHeader {
        self.width = width;
        self.height = height;
        self
    }
    /// Set the number of channels. Must be 3 or 4, for RGB or RGBA.
    /// 
    /// This must match the buffer:
    /// If a buffer of RGB pixels is encoded with an RGBA header, the encoding will simply fail.
    /// **If a buffer of RGBA pixels is encoded with an RGB header, encoding will succeed with a malformed output!**
    /// 
    /// # Panics
    /// 
    /// Function panics if `channels` is not 3 or 4
    pub fn channels(mut self, channels: u8) -> QOIHeader {
        if channels != 3 && channels != 4 {
            panic!("Invalid nr of channels");
        }
        self.channels = channels;
        self
    }
    /// Set the colorspace. Must be 0 or 1, for SRGB or all linear channels.
    /// 
    /// # Panics
    /// 
    /// Function panics if `colorspace` is not 0 or 1
    pub fn colorspace(mut self, colorspace: u8) -> QOIHeader {
        if colorspace != 0 && colorspace != 1 {
            panic!("Invalid colorspace");
        }
        self.colorspace = colorspace;
        self
    }

    fn to_be_bytes(&self) -> [u8; 14] {
        let mut out_bytes = [0; 14];
        MAGIC_BYTES.iter().enumerate().for_each(|(i, byte)| {out_bytes[i] = *byte}); //Write magic value
        self.width.to_be_bytes().iter().enumerate().for_each(|(i, byte)| {out_bytes[i+4] = *byte}); //Write width
        self.height.to_be_bytes().iter().enumerate().for_each(|(i, byte)| {out_bytes[i+8] = *byte}); //Write height
        out_bytes[12] = self.channels.to_be();
        out_bytes[13] = self.colorspace.to_be();
        out_bytes
    }
}


const QOI_OP_RUN  : u8 = 0b11000000;
const QOI_OP_DIFF : u8 = 0b01000000;
const QOI_OP_LUMA : u8 = 0b10000000;
const QOI_OP_RGB  : u8 = 0b11111110;
const QOI_OP_RGBA : u8 = 0b11111111;


#[derive(Clone, Copy, PartialEq)]
struct RGBA(u8, u8, u8, u8);

/// Returns a vector of bytes encoded in the QOI image format.
/// # Panics
/// Funtion panics if the buffer has fewer than `width * height * channels` bytes.
pub fn encode(buf: &[u8], header: QOIHeader) -> Vec<u8> {
    let img_length: usize = (header.width * header.height) as usize;
    let mut prev_pixel = RGBA(0,0,0,255);
    let mut seen_pixels = [RGBA(0,0,0,0); 64];

    let mut encoded_bytes = Vec::new();
    header.to_be_bytes().iter().for_each(|byte| {encoded_bytes.push(*byte)}); //Write header

    let mut op_run_count: u8 = 0; //For keeping track of run-length encoding between iterations
    let mut current_pixel = prev_pixel;
    for i in 0..img_length { //Iterates once per pixel
        prev_pixel = current_pixel;
        match header.channels {
            3 => {
                current_pixel = RGBA(buf[i*3], buf[i*3+1], buf[i*3+2], 255);
            },
            4 => {
                current_pixel = RGBA(buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]);
            },
            _ => unreachable!()
        }

        // Keep track of whether the pixel has been seen before
        let pixel_idx = index_hash_fn(current_pixel);
        let pixel_seen = if seen_pixels[pixel_idx as usize] == current_pixel {
            true
        } else {
            seen_pixels[pixel_idx as usize] = current_pixel;
            false
        };

        // QOI_OP_RUN -> Run length encoding, best strategy
        if current_pixel == prev_pixel { //Identical pixel
            if op_run_count < 62 { //If in the middle of a run
                op_run_count += 1; //Increment run length
                continue; //Skip to next iteration without writing anything   
            } else { //If we reach the limit for a chunk but it's still a run, write the chunk and continue
                encoded_bytes.push(QOI_OP_RUN | (op_run_count - 1).to_be());
                op_run_count = 1;
                continue;
            }
        } else if op_run_count > 0 { //If this is the end of a run, write the run-length chunk, then proceed with the other strategies for this pixel
            encoded_bytes.push(QOI_OP_RUN | (op_run_count - 1).to_be());
            op_run_count = 0;
        }

        // QOI_OP_INDEX -> Pixel is in the seen_pixels array
        if pixel_seen {
            encoded_bytes.push((pixel_idx as u8).to_be());
            continue;
        }

        if current_pixel.3 == prev_pixel.3 { //Alpha must match for DIFF, LUMA, and RGB

            // QOI_OP_DIFF
            {
            //Biased by 2, the range is [-2, 1] which becomes [0, 3]
            let dr = current_pixel.0.wrapping_add(2).wrapping_sub(prev_pixel.0);
            if dr < 4 {
                let dg = current_pixel.1.wrapping_add(2).wrapping_sub(prev_pixel.1);
                if dg < 4 {
                    let db = current_pixel.2.wrapping_add(2).wrapping_sub(prev_pixel.2);
                    if db < 4 {
                        encoded_bytes.push(QOI_OP_DIFF | (dr << 4).to_be() | (dg << 2).to_be() | db.to_be());
                        continue;
                    }
                }
            }    
            }

            // QOI_OP_LUMA
            {
            //Green biased by 32, red and blue by 8, biases added after calculating
            let dg = (current_pixel.1 as i8).wrapping_sub(prev_pixel.1 as i8);
            let dr_dg = ((current_pixel.0 as i8).wrapping_sub(prev_pixel.0 as i8)).wrapping_sub(dg);
            let db_dg = ((current_pixel.2 as i8).wrapping_sub(prev_pixel.2 as i8)).wrapping_sub(dg);
            if dg >= -32 && dg < 32 && dr_dg >= -8  && dr_dg < 8 && db_dg >= -8  && db_dg < 8 {
                encoded_bytes.push(QOI_OP_LUMA | ((dg + 32) as u8).to_be());
                encoded_bytes.push((((dr_dg + 8) as u8) << 4).to_be() | ((db_dg + 8) as u8).to_be());
                continue;            
            }
            }


            // If none of the other strategies are viable, write the pixel value directly
            //QOI_OP_RGB
            encoded_bytes.push(QOI_OP_RGB);
            encoded_bytes.push(current_pixel.0.to_be());
            encoded_bytes.push(current_pixel.1.to_be());
            encoded_bytes.push(current_pixel.2.to_be());
            continue;
        }
        //QOI_OP_RGBA
        encoded_bytes.push(QOI_OP_RGBA);
        encoded_bytes.push(current_pixel.0.to_be());
        encoded_bytes.push(current_pixel.1.to_be());
        encoded_bytes.push(current_pixel.2.to_be());
        encoded_bytes.push(current_pixel.3.to_be());
    }
    //If we run out of pixels mid-run, write the run-length chunk
    if op_run_count > 0 {
        encoded_bytes.push(QOI_OP_RUN | (op_run_count - 1).to_be());
    }

    // End the stream
    for _ in 0..7 {
        encoded_bytes.push( 0b00000000 );
    }
    encoded_bytes.push( 0b00000001 );

    encoded_bytes
}


/// Returns a vector of bytes decoded from the QOI image format if the file conforms to the format specification, error otherwise.
pub fn decode(buf: &[u8]) -> Result<(QOIHeader, Vec<u8>),()> {
    let mut decoded_bytes = Vec::new();
    let header_bytes = if let Some(v) = buf.get(0..14) {v} else {return Err(())};
    
    //Verify magic bytes
    let magic_bytes: [u8; 4] = header_bytes.get(0..4).unwrap().try_into().unwrap();
    if magic_bytes != *b"qoif" {
        return Err(())
    }
    //Verify tail bytes as well
    let tail_bytes: [u8; 8] = buf.get((buf.len()-8)..buf.len()).unwrap().try_into().unwrap();
    if u64::from_be_bytes(tail_bytes) != 1 {
        return Err(())
    }

    let header = QOIHeader {
        width: {
            let width_bytes = header_bytes.get(4..8).unwrap().try_into().unwrap();
            let width = u32::from_be_bytes(width_bytes);
            width
        },
        height: {
            let width_bytes = header_bytes.get(8..12).unwrap().try_into().unwrap();
            let width = u32::from_be_bytes(width_bytes);
            width
        },
        channels: u8::from_be(*header_bytes.get(12).unwrap()),
        colorspace: u8::from_be(*header_bytes.get(13).unwrap()),
    };

    

    let mut prev_pixel = RGBA(0,0,0,255);
    let mut current_pixel = prev_pixel;
    let mut seen_pixels = [RGBA(0,0,0,0); 64];

    let mut push_pixel = |pixel: RGBA| {
        decoded_bytes.push(pixel.0);
        decoded_bytes.push(pixel.1);
        decoded_bytes.push(pixel.2);
        if header.channels == 4 {
            decoded_bytes.push(pixel.3);
        }
    };

    let mut idx = 14; //Index of next chunk
    while idx < buf.len() - 8 { //Ignore tail bytes
        prev_pixel = current_pixel;
        let pix_idx = index_hash_fn(prev_pixel);
        seen_pixels[pix_idx] = prev_pixel;

        let chunk_head = buf[idx]; //Contains the tag indicating the qoi operation, might just be the entire chunk for 1-byte ops
        
        //QOI_OP_RGB
        if chunk_head == QOI_OP_RGB {
            current_pixel.0 = buf[idx+1];
            current_pixel.1 = buf[idx+2];
            current_pixel.2 = buf[idx+3];
            push_pixel(current_pixel);
            idx += 4;
            continue;
        }
        //QOI_OP_RGBA
        if chunk_head == QOI_OP_RGBA {
            current_pixel.0 = buf[idx+1];
            current_pixel.1 = buf[idx+2];
            current_pixel.2 = buf[idx+3];
            current_pixel.3 = buf[idx+4];
            push_pixel(current_pixel);
            idx += 5;
            continue;
        }


        //First two bits of each operation is:
        // QOI_OP_RUN = 11 = 3
        // QOI_OP_INDEX = 00 = 0
        // QOI_OP_DIFF = 01 = 1
        // QOI_OP_LUMA = 10 = 2

        //Check first 2 bits for tag
        match first_two_bits(chunk_head) {
            3 => { //QOI_OP_RUN
                let run_length = last_six_bits(chunk_head) + 1;
                for _ in 0..run_length {
                    push_pixel(prev_pixel);
                }
                idx += 1;
                continue;
            },
            0 => { //QOI_OP_INDEX
                let pix_idx = last_six_bits(chunk_head);
                current_pixel = seen_pixels[pix_idx as usize];
                push_pixel(current_pixel);
                idx += 1;
                continue;
            }
            1 => { //QOI_OP_DIFF
                let dr = ((u8::from_be(chunk_head) << 2) >> 6) as i16 - 2;
                let dg = ((u8::from_be(chunk_head) << 4) >> 6) as i16 - 2;
                let db = ((u8::from_be(chunk_head) << 6) >> 6) as i16 - 2;

                current_pixel.0 = (prev_pixel.0 as i16 + dr) as u8;
                current_pixel.1 = (prev_pixel.1 as i16 + dg) as u8;
                current_pixel.2 = (prev_pixel.2 as i16 + db) as u8;
                push_pixel(current_pixel);
                idx += 1;
                continue;
            },
            2 => { //QOI_OP_LUMA
                let dg = last_six_bits(chunk_head) as i16 - 32;

                let next_byte = buf[idx+1];
                let dr_dg = (u8::from_be(next_byte) >> 4) as i16 - 8;
                let db_dg = ((u8::from_be(next_byte) << 4) >> 4) as i16 - 8;

                let dr = dr_dg + dg;
                let db = db_dg + dg;

                current_pixel.0 = (prev_pixel.0 as i16 + dr) as u8;
                current_pixel.1 = (prev_pixel.1 as i16 + dg) as u8;
                current_pixel.2 = (prev_pixel.2 as i16 + db) as u8;
                push_pixel(current_pixel);
                idx += 2;
                continue;
            }
            _ => unreachable!()
        }

    }

    Ok((header, decoded_bytes))
}

fn index_hash_fn(pix: RGBA) -> usize {
    (pix.0.wrapping_mul(3).wrapping_add(
        pix.1.wrapping_mul(5)).wrapping_add(
        pix.2.wrapping_mul(7)).wrapping_add(
        pix.3.wrapping_mul(11)) % 64) as usize
}

fn first_two_bits(num: u8) -> u8 {
    u8::from_be(num) >> 6
}
fn last_six_bits(num: u8) -> u8 {
    (u8::from_be(num) << 2) >> 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Write, Read};
    use std::fs;
    use image;

    #[test]
    fn test_encoding() {
        let in_image = image::open("small_test.png").unwrap().into_rgba8();
        let header = QOIHeader::new().size(in_image.width(), in_image.height()).channels(4).colorspace(0);
        let mut out_file = fs::File::create("small_test.qoi").unwrap();

        let test_output = encode( in_image.as_flat_samples().as_slice() , header);
        
        out_file.write_all(&test_output).unwrap();
    }

    #[test]
    fn test_decoding() {
        let in_file = fs::File::open("small_test.qoi").unwrap();
        let in_bytes: Vec<u8> = in_file.bytes().map(|res| {res.unwrap()}).collect();

        let (header, decoded_bytes) = decode(&in_bytes).unwrap();
        
        let layout = image::flat::SampleLayout::row_major_packed(header.channels, header.width, header.height);
        let flat_samples = image::flat::FlatSamples {
            layout,
            samples: decoded_bytes,
            color_hint: None
        };

        let image_buffer: image::RgbaImage = flat_samples.try_into_buffer().unwrap();

        image_buffer.save("small_test_decoded.png").unwrap();
    }
}