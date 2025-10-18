import os

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

GEMINI_API_KEYS = [
    os.getenv(f'GEMINI_API_KEY_{i}', '') for i in range(1, 11)
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]

GEMINI_RPM_LIST = [
    int(os.getenv(f'GEMINI_RPM_{i}', '100')) for i in range(1, 11)
][:len(GEMINI_API_KEYS)]

GEMINI_RPD_LIST = [
    int(os.getenv(f'GEMINI_RPD_{i}', '10000')) for i in range(1, 11)
][:len(GEMINI_API_KEYS)]

if len(set(GEMINI_RPM_LIST)) == 1 and GEMINI_RPM_LIST:
    GEMINI_RPM_LIST = GEMINI_RPM_LIST[0]

if len(set(GEMINI_RPD_LIST)) == 1 and GEMINI_RPD_LIST:
    GEMINI_RPD_LIST = GEMINI_RPD_LIST[0]

MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', '30000'))
MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '4096'))

if __name__ == "__main__":
    print("Loaded Gemini API Keys:", len(GEMINI_API_KEYS))
    for idx, key in enumerate(GEMINI_API_KEYS, start=1):
        print(f"- Key {idx}: {key[:10]}...")
    print("RPM config:", GEMINI_RPM_LIST)
    print("RPD config:", GEMINI_RPD_LIST)
