from collections import deque

def lempel_ziv_encode(text):
    letters = deque([l for l in text])
    phrases = {}
    encoded = ""
    while len(letters) > 0:
        phrase = letters.popleft()
        while len(letters) > 0 and phrase in phrases:
            phrase += letters.popleft()
        phrases[phrase] = len(phrases) + 1
        if len(phrase) == 1:
            encoded += "0"
            encoded += phrase
        else:
            encoded += str(phrases[phrase[:-1]])
            encoded += phrase[-1]
    return encoded


if __name__ == '__main__':
    x = lempel_ziv_encode("ABAABABABBBAB")
    print(x)


