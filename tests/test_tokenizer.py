import json
import random
import tempfile
import unittest
from pathlib import Path

from tokenizer.bpe import GPT2_SPLIT_PATTERN, ByteBPETokenizer, count_pairs, merge_pair, pre_split
from tokenizer.documents import read_documents


class PreSplitTests(unittest.TestCase):
    def test_gpt2_style_chunks(self):
        self.assertEqual(
            pre_split("Hello world, it's 2026!", GPT2_SPLIT_PATTERN),
            ["Hello", " world", ",", " it", "'s", " 2026", "!"],
        )

    def test_whitespace_run_leaves_one_space_for_next_word(self):
        self.assertEqual(pre_split("a   b\n\n", GPT2_SPLIT_PATTERN), ["a", "  ", " b", "\n\n"])

    def test_underscore_and_contractions(self):
        self.assertEqual(pre_split("snake_case", GPT2_SPLIT_PATTERN), ["snake", "_", "case"])
        self.assertEqual(pre_split("WE'LL", GPT2_SPLIT_PATTERN), ["WE", "'LL"])

    def test_chunks_join_back_to_original(self):
        rng = random.Random(0)
        # Random code points from ASCII, Latin, Devanagari, CJK and emoji ranges,
        # skipping surrogates, which are not valid in UTF-8 text.
        ranges = [(0, 0x7F), (0xA0, 0x24F), (0x900, 0x97F), (0x4E00, 0x4FFF), (0x1F300, 0x1F64F)]
        samples = ["", " ", "नमस्ते 👋 café", "x́", "a\t\r\n b", "½ ² ٣"]
        for _ in range(200):
            samples.append("".join(chr(rng.randint(*rng.choice(ranges))) for _ in range(rng.randint(1, 40))))
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual("".join(pre_split(text, GPT2_SPLIT_PATTERN)), text)

    def test_no_pattern_keeps_text_whole(self):
        self.assertEqual(pre_split("a b", None), ["a b"])
        self.assertEqual(pre_split("", None), [])

    def test_merges_never_cross_chunks(self):
        corpus = ["ab.ab.ab.ab.ab"]
        split = ByteBPETokenizer.train(corpus, 270, pattern=GPT2_SPLIT_PATTERN)
        whole = ByteBPETokenizer.train(corpus, 270, pattern=None)
        learned = lambda tokenizer: [tokenizer.vocab[i] for i in range(259, tokenizer.vocab_size)]
        self.assertTrue(all(b"." not in token for token in learned(split)))
        self.assertTrue(any(b"." in token for token in learned(whole)))

    def test_load_version_1_file_without_pre_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({
                "format": "educational-byte-bpe",
                "version": 1,
                "special_tokens": {"<bos>": 256, "<eos>": 257, "<pad>": 258},
                "merges": [[98, 46]],
            }))
            tokenizer = ByteBPETokenizer.load(path)
            self.assertIsNone(tokenizer.pattern)
            self.assertEqual(tokenizer.encode("ab."), [97, 259])


class FastTrainerTests(unittest.TestCase):
    """The fast trainer must learn exactly what the simple reference learns."""

    def assert_same_training(self, texts, vocab_size, **kwargs):
        fast = ByteBPETokenizer.train(texts, vocab_size, **kwargs)
        reference = ByteBPETokenizer.train_reference(texts, vocab_size, **kwargs)
        self.assertEqual(fast.merges, reference.merges)
        self.assertEqual(fast.training_history, reference.training_history)

    def test_matches_reference_on_tiny_corpus(self):
        text = (Path(__file__).parent.parent / "examples" / "tiny_corpus.txt").read_text(encoding="utf-8")
        for pattern in (GPT2_SPLIT_PATTERN, None):
            with self.subTest(pattern=pattern):
                self.assert_same_training([text], 400, pattern=pattern)

    def test_matches_reference_on_random_corpora(self):
        rng = random.Random(1)
        for trial in range(60):
            # Few distinct characters give many repeats, ties and overlapping runs like "aaaa".
            alphabet = rng.choice(["ab", "abc ", "aab .", "xyz\n", "ab😀 "])
            texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
                     for _ in range(rng.randint(1, 5))]
            with self.subTest(trial=trial, texts=texts):
                self.assert_same_training(texts, rng.randint(259, 320),
                                          min_frequency=rng.randint(1, 3),
                                          pattern=rng.choice([GPT2_SPLIT_PATTERN, None]))

    def test_accepts_a_generator(self):
        tokenizer = ByteBPETokenizer.train((text for text in ["banana banana", "bandana"]), 270)
        self.assertEqual(tokenizer.merges, ByteBPETokenizer.train(["banana banana", "bandana"], 270).merges)


class FastEncodingTests(unittest.TestCase):
    TEXTS = ["the cat sat on the mat. The cat!", "", "नमस्ते 👋 café\n\n  x_y", "aaaa aaaa"]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ByteBPETokenizer.train(["the cat sat on the mat " * 5, "aaaa aaaa café"], 300)

    def uncached_encode(self, text):
        return [i for chunk in pre_split(text, self.tokenizer.pattern)
                for i in self.tokenizer.encode_bytes(chunk.encode("utf-8"))]

    def test_cache_gives_same_ids_and_stays_bounded(self):
        tokenizer = ByteBPETokenizer(self.tokenizer.merges)
        tokenizer.CHUNK_CACHE_SIZE = 3
        for _ in range(2):  # The second pass reads from the cache.
            for text in self.TEXTS:
                with self.subTest(text=text):
                    self.assertEqual(tokenizer.encode(text), self.uncached_encode(text))
                    self.assertLessEqual(len(tokenizer._chunk_cache), 3)

    def test_encode_many_matches_encode_in_order(self):
        expected = [self.tokenizer.encode(text, add_eos=True) for text in self.TEXTS * 10]
        for processes in (1, 2):
            with self.subTest(processes=processes):
                result = list(self.tokenizer.encode_many(self.TEXTS * 10, add_eos=True, processes=processes))
                self.assertEqual(result, expected)


class ReadDocumentsTests(unittest.TestCase):
    def test_jsonl_lines_and_plain_files(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "docs.jsonl"
            jsonl.write_text('{"id": 1, "text": "first"}\n\n{"id": 2, "text": "second\\nline"}\n', encoding="utf-8")
            plain = Path(directory) / "whole.txt"
            plain.write_text("one document\n", encoding="utf-8")
            self.assertEqual(list(read_documents([jsonl, plain])), ["first", "second\nline", "one document\n"])


class TokenizerTests(unittest.TestCase):
    def test_all_bytes_round_trip(self):
        raw = bytes(range(256))
        tokenizer = ByteBPETokenizer()
        self.assertEqual(tokenizer.decode_bytes(tokenizer.encode_bytes(raw)), raw)

    def test_unicode_and_whitespace_round_trip(self):
        tokenizer = ByteBPETokenizer.train(["hello hello", "banana banana"], 280)
        for text in ["", "hello", " hello\t\n  ", "नमस्ते 👋 café", "<eos>", "\x00"]:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_learning_compresses_repetition(self):
        tokenizer = ByteBPETokenizer.train(["banana banana banana"], 275)
        self.assertLess(len(tokenizer.encode("banana banana")), len(b"banana banana"))

    def test_document_boundaries(self):
        self.assertEqual(count_pairs([[97], [98]]), {})
        self.assertEqual(ByteBPETokenizer.train(["a", "b"], 280).merges, [])

    def test_overlap_is_merged_left_to_right(self):
        self.assertEqual(merge_pair([97, 97, 97], (97, 97), 259), [259, 97])
        tokenizer = ByteBPETokenizer.train(["aaaa"], 260)
        self.assertEqual(tokenizer.encode("aaaa"), [259, 259])

    def test_earlier_merge_rank_wins(self):
        tokenizer = ByteBPETokenizer([(97, 98), (98, 99)])
        self.assertEqual(tokenizer.encode("abc"), [259, 99])

    def test_deterministic_frequency_ties(self):
        first = ByteBPETokenizer.train(["ab", "cd"], 261, min_frequency=1)
        second = ByteBPETokenizer.train(["cd", "ab"], 261, min_frequency=1)
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.merges[0], (97, 98))

    def test_special_tokens_are_explicit(self):
        tokenizer = ByteBPETokenizer()
        ids = tokenizer.encode("<eos>", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tokenizer.BOS_ID)
        self.assertEqual(ids[-1], tokenizer.EOS_ID)
        self.assertEqual(ids[1:-1], list(b"<eos>"))
        self.assertEqual(tokenizer.decode(ids), "<eos>")
        with self.assertRaises(ValueError):
            tokenizer.decode(ids, skip_special_tokens=False)

    def test_save_reload_preserves_encoding(self):
        tokenizer = ByteBPETokenizer.train(["banana banana hello hello"], 275)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)
            self.assertEqual(tokenizer.merges, restored.merges)
            self.assertEqual(tokenizer.pattern, restored.pattern)
            self.assertEqual(tokenizer.encode("hello 👋"), restored.encode("hello 👋"))

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            ByteBPETokenizer.train("banana")
        with self.assertRaises(ValueError):
            ByteBPETokenizer.train([], 280)
        with self.assertRaises(ValueError):
            ByteBPETokenizer.train(["hello"], 258)
        with self.assertRaises(ValueError):
            ByteBPETokenizer([(256, 97)])
        with self.assertRaises(ValueError):
            ByteBPETokenizer().decode([999])


if __name__ == "__main__":
    unittest.main()
