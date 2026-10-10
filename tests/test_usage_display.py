"""Display metrics use provider input weights and one current context window."""
import unittest
from agent.storage.usage import summarize


class UsageDisplayTests(unittest.TestCase):
    def test_cache_weighted_by_input_and_context_not_cumulative(self):
        state = {'model_usage': [
            {'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                       'input_token_details': {'cache_read': 80}}},
            {'usage': {'input_tokens': 900, 'output_tokens': 40, 'total_tokens': 940,
                       'input_token_details': {'cache_read': 180}}}],
            'context_usage': {'used_tokens': 940, 'window_tokens': 1000000, 'estimated': False}}
        result = summarize(state)
        self.assertEqual(result['cache_rate'], .26)
        self.assertEqual(result['context']['used_tokens'], 940)
        self.assertEqual(result['total_tokens'], 1060)

    def test_missing_cache_and_context_are_unknown(self):
        result = summarize({'model_usage': [{'usage': {'input_tokens': 100,
            'output_tokens': 20, 'total_tokens': 120}}, {'usage': None}]})
        self.assertIsNone(result['cache_rate'])
        self.assertIsNone(result['context'])
        self.assertEqual(result['cache_unknown_calls'], 2)
        self.assertEqual(result['unknown_calls'], 1)
