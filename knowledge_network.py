import os
import json
import time
import re
import nltk
from typing import Dict, List, Tuple, Set
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

nltk.download('punkt')

class TagMatcher:
    def __init__(self, tags_directory: str):
        self.stemmer = PorterStemmer()
        self.tags = self._load_existing_tags(tags_directory)
        self.tag_stems = {tag: self.stemmer.stem(tag) for tag in self.tags}

    def _load_existing_tags(self, tags_directory: str) -> Set[str]:
        if not os.path.exists(tags_directory):
            return set()
        return {folder.lower() for folder in os.listdir(tags_directory) if os.path.isdir(os.path.join(tags_directory, folder))}

    def find_matching_tags(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        query_words = word_tokenize(query.lower())
        query_stems = [self.stemmer.stem(word) for word in query_words]

        matches = []
        for tag, tag_stem in self.tag_stems.items():
            if tag_stem in query_stems:
                matches.append((tag, 1.0))
                continue

            max_similarity = max(
                SequenceMatcher(None, tag_stem, query_stem).ratio()
                for query_stem in query_stems
            )

            if max_similarity >= threshold:
                matches.append((tag, max_similarity))

        return sorted(matches, key=lambda x: x[1], reverse=True)

class EnhancedDocumentProcessor:
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.tag_matcher = TagMatcher(os.path.join(db_dir, 'tags'))
        self.embeddings = OllamaEmbeddings(
            model="olmo2",
            base_url="http://localhost:11434"
        )
        self.llm = OllamaLLM(
            model="olmo2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        # Initialize vector store as Chroma instance
        self.vector_store = None
        main_store_path = os.path.join(db_dir, 'main')
        if os.path.exists(main_store_path):
            try:
                self.vector_store = Chroma(
                    persist_directory=main_store_path,
                    embedding_function=self.embeddings
                )
                print(f"Successfully loaded main vector store from {main_store_path}")
                # Print the number of documents in the store for debugging
                print(f"Number of documents in main store: {self.vector_store._collection.count()}")
            except Exception as e:
                print(f"Error loading main vector store: {str(e)}")

        self.tag_stores = self._load_existing_tag_stores()
        self.quizzes_dir = os.path.join(self.db_dir, "quizzes")
        self.lstm_data_file = os.path.join(self.db_dir, "lstm_training_data.json")

        if not os.path.exists(self.quizzes_dir):
            os.makedirs(self.quizzes_dir)

    def _load_existing_tag_stores(self) -> Dict[str, Chroma]:
        tag_stores = {}
        tags_dir = os.path.join(self.db_dir, 'tags')

        if not os.path.exists(tags_dir):
            return tag_stores
            
        for tag in os.listdir(tags_dir):
            tag_dir = os.path.join(tags_dir, tag)
            if os.path.isdir(tag_dir):
                try:
                    tag_stores[tag] = Chroma(
                        persist_directory=tag_dir,
                        embedding_function=self.embeddings
                    )
                except Exception as e:
                    print(f"Error loading tag store for {tag}: {str(e)}")

        return tag_stores

    def query_content(self, user_query: str) -> Dict:
        matching_tags = self.tag_matcher.find_matching_tags(user_query)
        
        if not matching_tags:
            if self.vector_store is not None:
                try:
                    print(f"Searching main vector store for: {user_query}")
                    
                    # Configure retriever with supported parameters
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": 1  # Number of documents to retrieve
                        }
                    )
                    
                    relevant_docs = retriever.get_relevant_documents(user_query)
                    print(f"Found {len(relevant_docs)} relevant documents")
                    
                    if relevant_docs:
                        print("First document content preview:", relevant_docs[0].page_content[:200])
                        
                        # Create QA chain with the retrieved documents
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=self.llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            verbose=True
                        )
                        
                        # Run the query
                        response = qa_chain({"query": user_query})
                        
                        # Check if we got a meaningful response
                        if response and 'result' in response:
                            return {
                                "type": "vector_store",
                                "response": response,
                                "num_docs": len(relevant_docs)
                            }
                        else:
                            print("No meaningful response from QA chain")
                            return {
                                "type": "generic",
                                "response": self.llm.invoke(user_query)
                            }
                    else:
                        print("No relevant documents found in vector store")
                        return {
                            "type": "generic",
                            "response": self.llm.invoke(user_query)
                        }
                        
                except Exception as e:
                    print(f"Error querying main vector store: {str(e)}")
                    return {
                        "type": "generic",
                        "response": self.llm.invoke(user_query)
                    }
            else:
                print("Main vector store is not initialized")
                return {
                    "type": "generic",
                    "response": self.llm.invoke(user_query)
                }
        
        # Handle case where tags were found
        for tag, similarity in matching_tags:
            if tag in self.tag_stores:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.tag_stores[tag].as_retriever(search_kwargs={"k": 1}),
                    return_source_documents=True
                )
                return {
                    "type": "tag",
                    "tag": tag,
                    "similarity": similarity,
                    "response": qa_chain({"query": user_query})
                }
        
        # If we get here, we had matching tags but no corresponding stores
        return {
            "type": "generic",
            "response": self.llm.invoke(user_query)
        }

    def generate_quiz(self, prompt: str) -> None:   
        try:
            num_questions = int(input("Enter the number of quiz questions: ").strip())
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
        
        quiz_data = []
        existing_questions = set()  # Store unique questions to avoid duplicates

        for i in range(1, num_questions + 1):
            print(f"Generating question {i}/{num_questions}... Please wait.")

            while True:
                # Modify prompt to request unique questions
                quiz_prompt = f"Create a **unique** quiz question on {prompt}. Do not repeat any previously asked question. " \
                            "Use the following structure, and **do not add any other text** except the dictionary format: " \
                            '{"question": "The question to ask", "choices": ["option1", "option2", "option3", "option4"], "correct_answer": "option3"}'

                response = self.llm.invoke(quiz_prompt)

                # Use regex to extract JSON structure
                matches = re.findall(r'\{.*?\}', response, re.DOTALL)
                
                if matches:
                    try:
                        question_data = json.loads(matches[0])  # Extract first valid question
                        question_text = question_data.get("question", "").strip()

                        if question_text and question_text not in existing_questions:
                            existing_questions.add(question_text)  # Store question to prevent duplicates
                            quiz_data.append(question_data)
                            print(f"✔ Question {i} generated successfully.")
                            break
                        else:
                            print("Duplicate question detected. Retrying...")
                    except json.JSONDecodeError:
                        print("Error decoding AI response. Retrying...")

        # Save all questions in the same file
        file_path = os.path.join(self.quizzes_dir, f"{prompt.replace(' ', '_')}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=4)

        print(f"\n✅ Quiz saved successfully: {file_path}")

    def ask_quiz_questions(self, quiz_file_path: str) -> None:
        # Load quiz questions from the specified file
        try:
            with open(quiz_file_path, "r", encoding="utf-8") as file:
                quiz_data = json.load(file)
        except FileNotFoundError:
            print("Quiz file not found.")
            return
        except json.JSONDecodeError:
            print("Error reading the quiz file.")
            return

        # Track overall interaction data
        interaction_data_list = []

        # Loop over each question in the quiz
        for question_data in quiz_data:
            question = question_data.get("question")
            choices = question_data.get("choices")
            correct_answer = question_data.get("correct_answer")

            if not all([question, choices, correct_answer]):
                print("Incomplete question data.")
                continue

            print(f"\nQuestion: {question}")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")

            start_time = time.time()
            user_answers = input("Select your answers (comma separated numbers): ").strip()
            user_answers = [int(i) - 1 for i in user_answers.split(',')]  # Convert to 0-indexed

            time_spent = time.time() - start_time

            # Determine correct answers
            correct_answers = [i for i, choice in enumerate(choices) if choice == correct_answer]

            # Track if the user answered correctly
            correct_answers_given = [choices[i] for i in user_answers if i in correct_answers]

            interaction_data = {
                "query": question,
                "response": question_data,
                "time_spent": time_spent,
                "correct_answers": correct_answers_given,
                "timestamp": time.time()
            }
            
            interaction_data_list.append(interaction_data)

        # Save interaction data for further analysis
        interaction_data_file = "user_interactions.json"
        try:
            if os.path.exists(interaction_data_file):
                with open(interaction_data_file, "r", encoding="utf-8") as f:
                    all_interactions = json.load(f)
            else:
                all_interactions = []
            
            all_interactions.extend(interaction_data_list)

            with open(interaction_data_file, "w", encoding="utf-8") as f:
                json.dump(all_interactions, f, indent=4)

            print(f"Interaction data saved successfully to {interaction_data_file}")
        
        except Exception as e:
            print(f"Error saving interaction data: {e}")

def main():
    db_dir = "database"
    processor = EnhancedDocumentProcessor(db_dir)

    while True:
        print("\nOptions:")
        print("1. Ask a normal prompt")
        print("2. Generate a quiz")
        print("3. Quiz time")
        print("4. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            query = input("Enter your query: ").strip()
            response = processor.query_content(query)
            print("Response:", response["response"])
        elif choice == '2':
            topic = input("Enter topic for quiz: ").strip()
            processor.generate_quiz(topic)
            # Specify the quiz file you want to load
            
        elif choice == '3':
            quiz_file_path = "database/quizzes/r_language.json"            
            # Ask the quiz questions and track interaction data
            processor.ask_quiz_questions(quiz_file_path)
            return

        elif choice == '4':
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
