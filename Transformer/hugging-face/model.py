from transformers import AutoTokenizer, AutoModel

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    text = "Hugging Face makes working with Transformers easy!"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    print(outputs.last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)


if __name__=="__main__":
    main()