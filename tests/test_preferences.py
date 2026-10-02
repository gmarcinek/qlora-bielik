from bielik_lora.preferences import chunk_text, dpo_record, preference_messages

CHAPTER = "Rozdzia\u0142 1\n\n" + "\n\n".join(f"Akapit {index} " + "s\u0142owo " * 60 for index in range(5))
NEXT = "Rozdzia\u0142 2\n\n" + "Kr\u00f3tki akapit drugiego rozdzia\u0142u, kt\u00f3ry ma ponad dwie\u015bcie znak\u00f3w. " * 4


def test_chunks_respect_limit_and_chapters():
    chunks = chunk_text(CHAPTER + "\n\n" + NEXT, max_chars=900)
    assert all(len(chunk) <= 900 * 1.5 for chunk in chunks)
    assert chunks[0].startswith("Rozdzia\u0142 1")
    assert any(chunk.startswith("Rozdzia\u0142 2") for chunk in chunks)
    assert not any("Rozdzia\u0142 1" in chunk and "Rozdzia\u0142 2" in chunk for chunk in chunks)


def test_long_paragraph_is_split_at_sentences():
    paragraph = " ".join(f"Zdanie numer {index} ko\u0144czy si\u0119 kropk\u0105." for index in range(200))
    chunks = chunk_text(paragraph, max_chars=500)
    assert len(chunks) > 1 and all(chunk.endswith(".") for chunk in chunks)


def test_dpo_record_has_trl_conversational_shape():
    messages = preference_messages("Napisz scen\u0119.", "Oryginalny fragment.", system="Styl X")
    record = dpo_record(messages, "Wersja modelu.")
    assert [message["role"] for message in record["prompt"]] == ["system", "user"]
    assert record["chosen"] == [{"role": "assistant", "content": "Oryginalny fragment."}]
    assert record["rejected"] == [{"role": "assistant", "content": "Wersja modelu."}]
