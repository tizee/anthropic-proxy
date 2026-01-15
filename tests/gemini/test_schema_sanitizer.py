import unittest

from anthropic_proxy.converter import clean_gemini_schema


class TestGeminiSchemaSanitizer(unittest.TestCase):
    def test_clean_gemini_schema_edge_cases(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "enum": [1, 2]},
                "items": {"type": "array"},
            },
            "required": ["count", "missing"],
        }

        cleaned = clean_gemini_schema(schema)
        self.assertEqual(cleaned["properties"]["count"]["type"], "string")
        self.assertEqual(cleaned["properties"]["count"]["enum"], ["1", "2"])
        self.assertEqual(cleaned["properties"]["items"]["items"], {})
        self.assertEqual(cleaned["required"], ["count"])

    def test_clean_gemini_schema_strips_code_assist_unsupported_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number", "exclusiveMinimum": 0},
                "name": {"type": "string"},
            },
            "propertyNames": {"pattern": "^[a-z]+$"},
        }

        cleaned = clean_gemini_schema(schema)

        self.assertNotIn("exclusiveMinimum", cleaned["properties"]["value"])
        self.assertNotIn("propertyNames", cleaned)
        self.assertIn("exclusiveMinimum: 0", cleaned["properties"]["value"]["description"])

    def test_clean_gemini_schema_normalizes_type_names(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "count": {"type": "INTEGER", "enum": [1, 2]},
                "items": {"type": "ARRAY"},
            },
        }

        cleaned = clean_gemini_schema(schema)

        self.assertEqual(cleaned["type"], "object")
        self.assertEqual(cleaned["properties"]["count"]["type"], "string")
        self.assertEqual(cleaned["properties"]["count"]["enum"], ["1", "2"])
        self.assertEqual(cleaned["properties"]["items"]["items"], {})


if __name__ == "__main__":
    unittest.main()
