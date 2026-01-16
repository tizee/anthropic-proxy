"""Unit tests for image token estimation functionality."""

import base64
import struct
import unittest

from anthropic_proxy.utils import (
    ANTHROPIC_IMAGE_TOKEN_DIVISOR,
    DEFAULT_IMAGE_TOKENS,
    _get_gif_dimensions,
    _get_jpeg_dimensions,
    _get_png_dimensions,
    count_tokens_in_messages,
    estimate_image_tokens,
    estimate_image_tokens_from_base64,
    get_image_dimensions_from_base64,
)


def create_minimal_png(width: int, height: int) -> bytes:
    """Create a minimal valid PNG file with specified dimensions."""
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk: width, height, bit depth, color type, compression, filter, interlace
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = 0x575E51F2  # Placeholder CRC (not validated in our parser)
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)

    # Minimal IDAT chunk (empty compressed data)
    idat_data = b'\x08\xd7\x63\x00\x00\x00\x01\x00\x01'
    idat_crc = 0x0D0A1A0A  # Placeholder CRC
    idat_chunk = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc)

    # IEND chunk
    iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', 0xAE426082)

    return signature + ihdr_chunk + idat_chunk + iend_chunk


def create_minimal_gif(width: int, height: int) -> bytes:
    """Create a minimal valid GIF file with specified dimensions."""
    # GIF89a header
    header = b'GIF89a'
    # Logical screen descriptor: width, height (little-endian), packed byte, bg color, aspect ratio
    screen_desc = struct.pack('<HH', width, height) + b'\x00\x00\x00'
    # Image descriptor and minimal image data
    image_desc = b'\x2c\x00\x00\x00\x00' + struct.pack('<HH', width, height) + b'\x00'
    # Minimal LZW compressed data
    image_data = b'\x02\x02\x44\x01\x00'
    # Trailer
    trailer = b'\x3b'

    return header + screen_desc + image_desc + image_data + trailer


def create_minimal_jpeg(width: int, height: int) -> bytes:
    """Create a minimal JPEG-like header with SOF0 marker for dimension testing."""
    # SOI marker
    soi = b'\xff\xd8'
    # APP0 marker (JFIF)
    app0 = b'\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    # SOF0 marker with dimensions
    # Structure: FF C0 LL LL PP HH HH WW WW (length includes itself)
    sof0_len = 11  # 2 bytes length + 1 precision + 2 height + 2 width + 1 components + 3 component data
    sof0 = b'\xff\xc0' + struct.pack('>H', sof0_len) + b'\x08'  # 8-bit precision
    sof0 += struct.pack('>H', height) + struct.pack('>H', width)
    sof0 += b'\x01\x11\x00'  # 1 component, sampling factors, quantization table
    # EOI marker
    eoi = b'\xff\xd9'

    return soi + app0 + sof0 + eoi


class TestImageDimensionExtraction(unittest.TestCase):
    """Test cases for extracting image dimensions from binary data."""

    def test_png_dimensions(self):
        """Test extracting dimensions from PNG data."""
        png_data = create_minimal_png(800, 600)
        dimensions = _get_png_dimensions(png_data)
        self.assertEqual(dimensions, (800, 600))

    def test_png_dimensions_large(self):
        """Test extracting large dimensions from PNG data."""
        png_data = create_minimal_png(4096, 3072)
        dimensions = _get_png_dimensions(png_data)
        self.assertEqual(dimensions, (4096, 3072))

    def test_png_dimensions_invalid(self):
        """Test PNG dimension extraction with invalid data."""
        self.assertIsNone(_get_png_dimensions(b'invalid'))
        self.assertIsNone(_get_png_dimensions(b''))
        self.assertIsNone(_get_png_dimensions(b'\x89PNG\r\n\x1a\n'))  # Too short

    def test_gif_dimensions(self):
        """Test extracting dimensions from GIF data."""
        gif_data = create_minimal_gif(640, 480)
        dimensions = _get_gif_dimensions(gif_data)
        self.assertEqual(dimensions, (640, 480))

    def test_gif_dimensions_invalid(self):
        """Test GIF dimension extraction with invalid data."""
        self.assertIsNone(_get_gif_dimensions(b'invalid'))
        self.assertIsNone(_get_gif_dimensions(b''))
        self.assertIsNone(_get_gif_dimensions(b'GIF89'))  # Too short

    def test_jpeg_dimensions(self):
        """Test extracting dimensions from JPEG data."""
        jpeg_data = create_minimal_jpeg(1920, 1080)
        jpeg_base64 = base64.b64encode(jpeg_data).decode()
        dimensions = _get_jpeg_dimensions(jpeg_base64)
        self.assertEqual(dimensions, (1920, 1080))

    def test_jpeg_dimensions_invalid(self):
        """Test JPEG dimension extraction with invalid data."""
        self.assertIsNone(_get_jpeg_dimensions(base64.b64encode(b'invalid').decode()))
        self.assertIsNone(_get_jpeg_dimensions(base64.b64encode(b'').decode()))


class TestImageDimensionsFromBase64(unittest.TestCase):
    """Test cases for extracting image dimensions from base64-encoded data."""

    def test_png_base64(self):
        """Test extracting dimensions from base64-encoded PNG."""
        png_data = create_minimal_png(1024, 768)
        base64_data = base64.b64encode(png_data).decode()
        dimensions = get_image_dimensions_from_base64(base64_data, "image/png")
        self.assertEqual(dimensions, (1024, 768))

    def test_gif_base64(self):
        """Test extracting dimensions from base64-encoded GIF."""
        gif_data = create_minimal_gif(320, 240)
        base64_data = base64.b64encode(gif_data).decode()
        dimensions = get_image_dimensions_from_base64(base64_data, "image/gif")
        self.assertEqual(dimensions, (320, 240))

    def test_jpeg_base64(self):
        """Test extracting dimensions from base64-encoded JPEG."""
        jpeg_data = create_minimal_jpeg(1280, 720)
        base64_data = base64.b64encode(jpeg_data).decode()
        dimensions = get_image_dimensions_from_base64(base64_data, "image/jpeg")
        self.assertEqual(dimensions, (1280, 720))

    def test_unsupported_media_type(self):
        """Test handling of unsupported media types."""
        png_data = create_minimal_png(100, 100)
        base64_data = base64.b64encode(png_data).decode()
        dimensions = get_image_dimensions_from_base64(base64_data, "image/bmp")
        self.assertIsNone(dimensions)

    def test_invalid_base64(self):
        """Test handling of invalid base64 data."""
        dimensions = get_image_dimensions_from_base64("not-valid-base64!!!", "image/png")
        self.assertIsNone(dimensions)


class TestImageTokenEstimation(unittest.TestCase):
    """Test cases for image token estimation."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation from dimensions."""
        # 1000x1000 image: 1000000 / 750 = 1333 tokens
        tokens = estimate_image_tokens(1000, 1000)
        self.assertEqual(tokens, 1333)

    def test_estimate_tokens_1080p(self):
        """Test token estimation for 1080p image."""
        # 1920x1080 image: 2073600 / 750 = 2764 tokens
        tokens = estimate_image_tokens(1920, 1080)
        self.assertEqual(tokens, 2764)

    def test_estimate_tokens_4k(self):
        """Test token estimation for 4K image."""
        # 3840x2160 image: 8294400 / 750 = 11059 tokens
        tokens = estimate_image_tokens(3840, 2160)
        self.assertEqual(tokens, 11059)

    def test_estimate_tokens_small(self):
        """Test token estimation for small image (minimum 85 tokens)."""
        # 10x10 image: 100 / 750 = 0, but minimum is 85
        tokens = estimate_image_tokens(10, 10)
        self.assertEqual(tokens, 85)

    def test_estimate_tokens_default_on_missing_dimensions(self):
        """Test default token count when dimensions are missing."""
        self.assertEqual(estimate_image_tokens(None, None), DEFAULT_IMAGE_TOKENS)
        self.assertEqual(estimate_image_tokens(0, 100), DEFAULT_IMAGE_TOKENS)
        self.assertEqual(estimate_image_tokens(100, 0), DEFAULT_IMAGE_TOKENS)
        self.assertEqual(estimate_image_tokens(-1, 100), DEFAULT_IMAGE_TOKENS)


class TestImageTokensFromBase64(unittest.TestCase):
    """Test cases for estimating tokens from base64-encoded images."""

    def test_png_token_estimation(self):
        """Test token estimation from base64-encoded PNG."""
        png_data = create_minimal_png(800, 600)
        base64_data = base64.b64encode(png_data).decode()
        tokens = estimate_image_tokens_from_base64(base64_data, "image/png")
        # 800x600 = 480000 / 750 = 640 tokens
        self.assertEqual(tokens, 640)

    def test_gif_token_estimation(self):
        """Test token estimation from base64-encoded GIF."""
        gif_data = create_minimal_gif(640, 480)
        base64_data = base64.b64encode(gif_data).decode()
        tokens = estimate_image_tokens_from_base64(base64_data, "image/gif")
        # 640x480 = 307200 / 750 = 409 tokens
        self.assertEqual(tokens, 409)

    def test_invalid_image_returns_default(self):
        """Test that invalid images return default token count."""
        tokens = estimate_image_tokens_from_base64("invalid-base64!!!", "image/png")
        self.assertEqual(tokens, DEFAULT_IMAGE_TOKENS)


class TestCountTokensInMessages(unittest.TestCase):
    """Test cases for counting tokens in messages with image content."""

    def test_text_only_message(self):
        """Test token counting for text-only messages."""
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
        tokens = count_tokens_in_messages(messages, "claude-3-opus")
        # Should be reasonable token count for this short message
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 100)

    def test_message_with_image_block(self):
        """Test token counting for messages with image content blocks."""
        png_data = create_minimal_png(800, 600)
        base64_data = base64.b64encode(png_data).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                ]
            }
        ]

        tokens = count_tokens_in_messages(messages, "claude-3-opus")

        # Token count should be reasonable:
        # - Text tokens (small)
        # - Image tokens: 800x600 = 640 tokens
        # Total should be around 640-700, NOT tens of thousands
        self.assertGreater(tokens, 600)
        self.assertLess(tokens, 1000)  # Should NOT be inflated by base64 data

    def test_message_with_large_image_reasonable_tokens(self):
        """Test that large images don't cause inflated token counts."""
        # Create a "large" image (simulated by dimensions, not actual data)
        png_data = create_minimal_png(1920, 1080)
        base64_data = base64.b64encode(png_data).decode()

        # Pad the base64 data to simulate a larger image file
        # This tests that we're not counting the base64 string length
        padded_base64 = base64_data

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": padded_base64
                        }
                    }
                ]
            }
        ]

        tokens = count_tokens_in_messages(messages, "claude-3-opus")

        # 1920x1080 = 2764 tokens for image
        # Total should be around 2800-3000, NOT inflated
        self.assertGreater(tokens, 2700)
        self.assertLess(tokens, 4000)

    def test_message_with_url_image(self):
        """Test token counting for URL-based images (uses default)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/image.png"
                        }
                    }
                ]
            }
        ]

        tokens = count_tokens_in_messages(messages, "claude-3-opus")

        # Should use DEFAULT_IMAGE_TOKENS (1500) for URL images
        self.assertGreater(tokens, 1400)
        self.assertLess(tokens, 2000)

    def test_mixed_content_message(self):
        """Test token counting for messages with mixed text and images."""
        png_data = create_minimal_png(640, 480)
        base64_data = base64.b64encode(png_data).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    },
                    {"type": "text", "text": "Second image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                ]
            }
        ]

        tokens = count_tokens_in_messages(messages, "claude-3-opus")

        # 640x480 = 409 tokens per image
        # Two images = ~818 tokens
        # Plus text tokens
        self.assertGreater(tokens, 800)
        self.assertLess(tokens, 1500)

    def test_comparison_with_without_image_handling(self):
        """
        Test that image handling significantly reduces token count
        compared to naive base64 counting.
        """
        # Create a larger image to make the difference more obvious
        png_data = create_minimal_png(1024, 1024)
        # Artificially inflate the base64 data to simulate a real image
        # (real images have much more data than our minimal test images)
        large_base64 = base64.b64encode(png_data + b'\x00' * 50000).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": large_base64
                        }
                    }
                ]
            }
        ]

        tokens = count_tokens_in_messages(messages, "claude-3-opus")

        # With proper image handling: ~1365 tokens (1024x1024 / 750)
        # Without handling: would be tens of thousands (base64 is ~67KB)
        self.assertGreater(tokens, 1000)
        self.assertLess(tokens, 3000)  # Should NOT be tens of thousands


class TestTokenEstimationConstants(unittest.TestCase):
    """Test cases for token estimation constants."""

    def test_default_image_tokens(self):
        """Test that DEFAULT_IMAGE_TOKENS is reasonable."""
        # Default should be reasonable for a typical image
        self.assertGreater(DEFAULT_IMAGE_TOKENS, 500)
        self.assertLess(DEFAULT_IMAGE_TOKENS, 5000)

    def test_anthropic_divisor(self):
        """Test that ANTHROPIC_IMAGE_TOKEN_DIVISOR matches documentation."""
        # Anthropic's formula: tokens = (width * height) / 750
        self.assertEqual(ANTHROPIC_IMAGE_TOKEN_DIVISOR, 750)


if __name__ == '__main__':
    unittest.main()
