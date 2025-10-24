# English-Grammar-and-speeling-checker
# ============================================================
# 🇬🇧 English NLP Spelling, Grammar & Root Word Checker
# ============================================================

# Install dependencies
!pip install pyspellchecker language-tool-python nltk textblob

# ------------------------------------------------------------
# 📚 Import & NLTK Downloads
# ------------------------------------------------------------
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # fix for new NLTK versions
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# ------------------------------------------------------------
# ✂️ TEXT CLEANER – Remove punctuation & extra spaces
# ------------------------------------------------------------
import re
import string

def clean_text(text):
    """Remove punctuation and unnecessary spaces from the input text."""
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------
# 🔤 SPELLING CHECKER (English)
# ------------------------------------------------------------
from spellchecker import SpellChecker

def check_spelling(text):
    """Check and correct English spelling errors."""
    spell = SpellChecker(language='en')
    words = text.split()
    misspelled = spell.unknown(words)
    corrections = {word: spell.correction(word) for word in misspelled}
    return corrections

# ------------------------------------------------------------
# 📝 GRAMMAR CHECKER (English only)
# ------------------------------------------------------------
import language_tool_python

def check_grammar(text):
    """Use LanguageTool Public API to find grammar issues in English."""
    try:
        tool = language_tool_python.LanguageToolPublicAPI('en-US')
        matches = tool.check(text)
        grammar_errors = []
        for match in matches:
            grammar_errors.append({
                "Error": match.message,
                "Suggested": match.replacements,
                "Incorrect Text": text[match.offset: match.offset + match.errorLength]
            })
        return grammar_errors
    except Exception as e:
        print(f"⚠️ Grammar check unavailable (API/network issue): {e}")
        return []

# ------------------------------------------------------------
# 🧠 NLP ANALYSIS (Tokenization, POS, Sentiment, Root Words)
# ------------------------------------------------------------
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer, PorterStemmer

def nlp_analysis(text):
    """Perform NLP analysis including tokenization, POS, sentiment, and root words."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    sentiment = TextBlob(text).sentiment

    # Root word extraction
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    roots = {
        token: {
            "stem": stemmer.stem(token),
            "lemma": lemmatizer.lemmatize(token)
        }
        for token in tokens
    }

    return tokens, pos_tags, sentiment, roots

# ------------------------------------------------------------
# ⚙️ MAIN FUNCTION (Runtime Input)
# ------------------------------------------------------------
def analyze_text(user_text):
    print("==== 🇬🇧 English NLP Spelling, Grammar & Root Word Checker ====")
    print(f"Original Input: {user_text}\n")

    # Step 1: Clean text
    cleaned_text = clean_text(user_text)
    print(f"🧹 Cleaned Text: {cleaned_text}\n")

    # Step 2: Grammar check
    grammar_issues = check_grammar(cleaned_text)
    if grammar_issues:
        print("🔍 Grammar Issues:")
        for issue in grammar_issues:
            print(f"  ❌ {issue['Error']}")
            print(f"     Incorrect: {issue['Incorrect Text']}")
            print(f"     Suggestions: {issue['Suggested']}")
            print("-" * 40)
    else:
        print("✅ No grammar issues found!")

    # Step 3: Spelling check
    spelling_corrections = check_spelling(cleaned_text)
    if spelling_corrections:
        print("\n📝 Spelling Suggestions:")
        for word, suggestion in spelling_corrections.items():
            print(f"  {word} → {suggestion}")
    else:
        print("\n✅ No spelling mistakes found!")

    # Step 4: NLP analysis
    print("\n🧠 NLP Analysis:")
    tokens, pos_tags, sentiment, roots = nlp_analysis(cleaned_text)
    print(f"Tokens: {tokens}")
    print(f"POS Tags: {pos_tags}")
    print(f"Sentiment: Polarity={sentiment.polarity:.2f}, Subjectivity={sentiment.subjectivity:.2f}")

    print("\n🌱 Root Word Extraction:")
    for word, forms in roots.items():
        print(f"  {word:15} → Stem: {forms['stem']:10} | Lemma: {forms['lemma']}")
    print("=" * 70)

# ------------------------------------------------------------
# 🚀 RUN THE PROGRAM
# ------------------------------------------------------------
user_text = input("📝 Enter your English text: ")
analyze_text(user_text)
