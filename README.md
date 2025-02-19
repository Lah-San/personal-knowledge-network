Personal Knowledge Network (PKN)
This application leverages a local RAG (Retrieval-Augmented Generation) framework to use your short notes for answering questions. It also incorporates an LSTM model to predict memory loss in specific study areas and provides personalized focus suggestions based on quiz results.

Features
Local RAG framework for context-based question answering using your notes.
LSTM model to track memory retention and predict memory loss.
Personalized suggestions for effective study focus based on quiz performance.
How It Works
Notes Processing: Upload your short notes (.txt, .pdf, .docx, or .html). The RAG framework embeds and stores them for context-based question answering.
Quiz Interaction: Answer quiz questions related to your notes. The system tracks time spent, correctness, and patterns.
Memory Prediction: An LSTM model uses quiz results and user interactions to predict memory retention for each topic.
Personalized Suggestions: Based on memory predictions, the system provides targeted study recommendations for areas that need more focus.
Requirements
Python 3.8 or later
PyTorch
LangChain
Ollama (for local LLM inference)
Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/personal-knowledge-network.git  
cd personal-knowledge-network  

# Create and activate a virtual environment
python -m venv venv  
source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

# Install dependencies
pip install -r requirements.txt  
Usage
Start Ollama Server:
Ensure Ollama is running locally:
bash
Copy
Edit
ollama start --model olmo2 --port 11434
Run the Application:
bash
Copy
Edit
python main.py  
Upload Notes:
Upload short notes in .txt, .pdf, .docx, or .html format.

Take Quizzes:
Answer quizzes generated based on your notes.

Get Recommendations:
View personalized focus suggestions based on quiz results and memory predictions.

Memory Loss Prediction
The LSTM model predicts memory retention starting from 100%. If incorrect or inconsistent answers are given, the memory score decreases. The application tracks memory scores over time and provides predictions for potential memory loss if suggested tasks are not completed.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

License
This project is licensed under the MIT License.
