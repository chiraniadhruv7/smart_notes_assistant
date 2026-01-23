from smart_notes.core.interfaces import LLM


class DummyLLM(LLM):
    def generate(self, prompt: str) -> str:
        return "LLM response placeholder.\n\nPrompt was:\n" + prompt
