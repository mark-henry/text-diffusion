from datasets import load_dataset

def download_and_inspect_dataset():
    """
    Downloads the WikiText-2 dataset and prints some information about it.
    """
    print("Downloading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset downloaded successfully!")
    print("\nDataset information:")
    print(dataset)

    print("\nLet's look at a few examples from the 'train' split:")
    train_dataset = dataset['train']
    for i in range(5):
        # The dataset contains a lot of empty lines, let's find non-empty examples
        example = train_dataset[i]
        text = example['text'].strip()
        if text:
            print(f"Example {i+1}:")
            print(text)
            print("-" * 20)

if __name__ == "__main__":
    download_and_inspect_dataset() 