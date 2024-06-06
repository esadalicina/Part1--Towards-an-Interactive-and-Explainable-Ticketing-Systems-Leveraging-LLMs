from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the MBart model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_text(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Example ticket text
ticket_text = "The server is down and needs to be restarted."

# Translate the ticket to German
translated_to_german = translate_text(ticket_text, 'en_XX', 'de_DE')
print("Translated to German:", translated_to_german)

# Translate the ticket to French
translated_to_french = translate_text(ticket_text, 'en_XX', 'fr_XX')
print("Translated to French:", translated_to_french)
