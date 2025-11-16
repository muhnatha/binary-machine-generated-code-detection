# Binary Code Classification of Human-Written or Machine-Generated
A deep learning project to classify code as human-written or machine-generated, targeting SemEval-2026 Task 13 subtask A.

## Requirements
1. Python version `3.10.x/3.11.x/3.12.x/3.13.x`

## How to Setup the Environment
1. Clone repository
   ```
   git clone https://github.com/muhnatha/binary-machine-generated-code-detection.git
   cd binary-machine-generated-code-detection
   ```
2. Create and activate python virtual environment
   ```
   python -m venv .venv

   # bash
   source .venv/scripts/activate

   # cmd/powershell
   .venv/scripts/activate
   ```
3. Install requirements
   ```
   pip install -r requirements.txt
   ```
4. Create .env file
5. Setup HuggingFace Token API
   ```
   # write in your .env
   HF_TOKEN = "your_api_token"
   ```
   
Shared Task: [https://github.com/mbzuai-nlp/SemEval-2026-Task13](https://github.com/mbzuai-nlp/SemEval-2026-Task13)