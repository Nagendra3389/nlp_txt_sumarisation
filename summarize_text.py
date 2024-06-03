import sys
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=500, min_length=250, do_sample=False):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, do_sample=do_sample, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_summary(file_path, summary):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summarize_text.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    summarizer = Summarizer()
    text = read_text(input_file)
    summary = summarizer.summarize(text)
    write_summary(output_file, summary)
    print(f"Summary written to {output_file}")