import os, json, tomllib
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path) and "GOOGLE_API_KEY" not in os.environ:
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        os.environ["GOOGLE_API_KEY"] = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY", "")

class CourseSpec(BaseModel):
    course_id: str = Field(description="e.g. aqa_gcse_combined_science_trilogy")
    course_title: str = Field(description="e.g. AQA GCSE Combined Science: Trilogy")
    level: str = Field(description="GCSE or A-Level")
    target_turns: int = Field(description="5 for GCSE, 7 for A-Level")
    topics: Dict[str, Dict[str, List[str]]] # Subject -> Unit -> Subtopics

def process_corpus():
    syllabus_dir = "./syllabus"
    pdf_files = sorted([f for f in os.listdir(syllabus_dir) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("⚠️ No PDFs found in ./syllabus/")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    for pdf in pdf_files:
        print(f"🔍 Processing {pdf}...")
        docs = PyPDFLoader(os.path.join(syllabus_dir, pdf)).load()
        
        # Keep full document for vector search, but sample headings/TOC across the whole PDF
        # We sample 1 page out of every 3 to get headings across all 198 pages without exceeding context limits
        sampled_pages = [docs[i].page_content for i in range(0, len(docs), 2)]
        full_overview_text = "\n--- PAGE ---\n".join(sampled_pages)

        print("   🧠 Extracting full course structure with Gemini...")
        prompt = (
            "You are analyzing an AQA GCSE Combined Science specification.\n"
            "Extract ALL main subject units and subtopics across BIOLOGY, CHEMISTRY, and PHYSICS.\n"
            "Ensure you do not stop early. Include Biology (e.g., Cell Biology, Organisation, Infection, Bioenergetics, Homeostasis, Inheritance, Ecology), "
            "Chemistry (Atomic structure, Bonding, Quantitative chemistry, Chemical changes, Energy changes, Rate of change, Organic chemistry, Analysis, Atmosphere, Using resources), "
            "and Physics (Energy, Electricity, Particle model, Atomic structure, Forces, Waves, Magnetism).\n\n"
            f"Specification Content Sample:\n{full_overview_text[:120000]}"
        )

        try:
            spec = llm.with_structured_output(CourseSpec).invoke(prompt)
            json_filename = f"{os.path.splitext(pdf)[0]}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(spec.model_dump(), f, indent=2)
            print(f"   ✅ Saved complete spec to {json_filename}!")
        except Exception as e:
            print(f"   ⚠️ Spec extraction error on {pdf}: {e}")

        # Vectorize all relevant content pages (pages 15 to end)
        content_docs = docs[15:] if len(docs) > 15 else docs
        chunks = splitter.split_documents(content_docs)
        for c in chunks:
            c.metadata["source_file"] = pdf
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            vectorstore.add_documents(chunks[i:i + batch_size])
        print(f"   ✅ Indexed {len(chunks)} chunks into ChromaDB.")

    print("\n🚀 All syllabus files processed successfully!")

if __name__ == "__main__":
    process_corpus()