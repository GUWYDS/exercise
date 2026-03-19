import fitz  # PyMuPDF
import re
import jieba
from typing import List, Tuple, Set

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from PDF and attempts to remove the References section.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # Basic cleaning: remove extra whitespace
        clean_text = " ".join(full_text.split())
        
        # Heuristic to remove References:
        # Looks for common headers like "References", "Bibliography", or "Literature Cited"
        # case-insensitive, usually at the start of a line or standing alone.
        ref_patterns = [
            r'\bReferences\b', 
            r'\bREFERENCES\b'
        ]
        
        # Find the last occurrence of these keywords
        split_index = len(clean_text)
        for pattern in ref_patterns:
            matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
            if matches:
                # We take the last match assuming References are at the end
                last_match_start = matches[-1].start()
                if last_match_start < split_index:
                    split_index = last_match_start
        
        if split_index < len(clean_text):
            print(f"--- Info: References detected and excluded for {pdf_path.split('/')[-1]}")
            return clean_text[:split_index]
            
        return clean_text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def get_ngram_list(text: str, n: int) -> List[Tuple[str, ...]]:
    """
    Tokenizes text and generates n-gram sequences.
    """
    # Tokenization: supports Chinese (jieba) and English (regex)
    tokens = [word for word in jieba.cut(text) if re.match(r'\w', word)]
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def merge_continuous_ngrams(ngram_list: List[Tuple[str, ...]], intersection: Set[Tuple[str, ...]]) -> List[str]:
    """
    Merges overlapping n-grams into continuous strings for better readability.
    """
    merged_sentences = []
    if not ngram_list:
        return merged_sentences

    current_sentence = []
    
    for i in range(len(ngram_list)):
        gram = ngram_list[i]
        
        if gram in intersection:
            if not current_sentence:
                current_sentence = list(gram)
            else:
                # Append the last word of the sliding window
                current_sentence.append(gram[-1])
        else:
            if current_sentence:
                merged_sentences.append(" ".join(current_sentence))
                current_sentence = []
    
    if current_sentence:
        merged_sentences.append(" ".join(current_sentence))
        
    return merged_sentences

def analyze_similarity(path1: str, path2: str, n_values=[5, 10], show_limit=15):
    """
    Main function to analyze and display similarity results.
    """
    print(f"Parsing files and stripping references...")
    text1 = extract_text_from_pdf(path1)
    text2 = extract_text_from_pdf(path2)
    
    if not text1 or not text2:
        print("Error: Could not extract enough text for comparison.")
        return

    print("\n" + "="*60)
    print("PLAGIARISM ANALYSIS REPORT")
    print("="*60)

    for n in n_values:
        list1 = get_ngram_list(text1, n)
        list2 = get_ngram_list(text2, n)
        
        set1, set2 = set(list1), set(list2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union) if union else 0
        
        print(f"\n>>> Dimension: {n}-gram")
        print(f"    Similarity Score: {similarity:.2%}")
        print(f"    Unique Matching n-grams: {len(intersection)}")
        
        if intersection:
            print(f"    Duplicate Snippets (Top {show_limit}, merged for readability):")
            merged = merge_continuous_ngrams(list1, intersection)
            
            # Remove duplicates from the display list
            unique_merged = []
            seen = set()
            for s in merged:
                if s not in seen:
                    unique_merged.append(s)
                    seen.add(s)

            for i, sentence in enumerate(unique_merged[:show_limit], 1):
                # Trim very long snippets
                display = (sentence[:150] + '...') if len(sentence) > 150 else sentence
                print(f"    [{i}] \"{display}\"")
        else:
            print("No significant overlaps found.")

if __name__ == "__main__":
    # Update paths as needed
    file_a = "/VisCom-HDD-1/wyf/D3/2405.04760v5.pdf"
    file_b = "/VisCom-HDD-1/wyf/D3/TOSEM-2026-0282_Proof_hi.pdf"
    
    analyze_similarity(file_a, file_b, n_values=[1,2,3,4,5,6,7,8,9,10,15], show_limit=20)
