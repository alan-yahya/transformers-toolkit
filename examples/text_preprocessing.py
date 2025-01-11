from transformer_toolkit import prepare_input

def preprocessing_example():
    """Demonstrate text cleaning and preprocessing."""
    # Example with messy text
    messy_text = """
    Check out this link: https://example.com
    Some special chars: @#$%^&*
    Multiple    spaces   and   lines...

    """
    
    # Clean text with different options
    clean_basic = prepare_input(messy_text, clean_text=True)
    clean_short = prepare_input(messy_text, max_length=50, clean_text=True)
    
    print("Original text:")
    print(messy_text)
    print("\nCleaned text:")
    print(clean_basic)
    print("\nCleaned and shortened text:")
    print(clean_short)

if __name__ == "__main__":
    preprocessing_example() 