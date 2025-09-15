# 🤖 DB Scope 

An intelligent observability platform that uses AI to automatically detect database performance bottlenecks, provide actionable optimization recommendations, and calculate potential cost savings.



## ✨ Key Features

- **Multi-Mode Analysis**:
  - **Auto-Detection (`auto`)**: Automatically identifies slow queries using native database tools (`pg_stat_statements` for Postgres, Performance Schema for MySQL).
  - **AI Health-Check (`benchmark`)**: Generates and runs a set of AI-created queries to proactively find performance weaknesses in your schema.
  - **File-Based (`file`)**: Allows users to upload a `.sql` file for direct analysis.
- **AI-Powered Suggestions**: Leverages GPT-5 to provide concrete optimization advice, including:
  - **Query Rewriting**: Modernizes legacy queries, optimizes joins, and replaces inefficient patterns.
  - **Intelligent Indexing**: Recommends `CREATE INDEX` statements for columns that would benefit most from indexing.
- **Interactive Chatbot**: A RAG-based conversational AI assistant that allows you to ask questions in natural language about your query, schema, and the AI's recommendations.
- **Verification Sandbox**: Safely execute original and optimized queries in a sandboxed environment to compare their results and verify that the optimization does not alter the outcome.
- **Cost & Performance Estimation**: Provides estimates on execution time improvements and potential cloud cost savings (e.g., for BigQuery, AWS Athena) by calculating the reduction in data scanned.
- **Multi-Database Support**: Built with an extensible architecture to support both **PostgreSQL** and **MySQL** out of the box.
- **Modern Frontend**: A clean, responsive dashboard built with Next.js, TypeScript, and shadcn/ui.

## ⚙️ Technical Stack

| Area          | Technology                                                                                                                                                                                             |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**   | Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/latest/), [SQLAlchemy](https://www.sqlalchemy.org/), [OpenAI API (GPT-5)](https://openai.com/)            |
| **Frontend**  | [Next.js](https://nextjs.org/)=, [React](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Tailwind CSS](https://tailwindcss.com/), [shadcn/ui](https://ui.shadcn.com/) |
| **Databases** | PostgreSQL, MySQL                                                                                                                                                                                      |

## 🏗️ Architecture

The project is architected as a monorepo with a decoupled frontend and backend.

### Backend Architecture

The backend follows a modular, service-oriented design pattern that is both scalable and extensible.

1.  **API Layer (`main.py`)**: A FastAPI server that exposes the REST API endpoints. It handles incoming HTTP requests, validates them using Pydantic models, and orchestrates the calls to the other modules.
2.  **Data Models (`models.py`)**: Pydantic models define the strict data contracts for all API requests and responses, ensuring type safety and clear documentation.
3.  **Database Inspector (`db_inspector.py`)**: An abstraction layer built using a **Factory Pattern**. The `DatabaseInspector` abstract base class defines a common interface for all database interactions. Concrete classes (`PostgresInspector`, `MySqlInspector`) implement this interface for specific SQL dialects. This makes the system easily extensible to new databases like SQL Server or SQLite.
4.  **AI Analyzer (`ai_analyzer.py`)**: This module encapsulates all interactions with the OpenAI API. It handles:
    - **Prompt Engineering**: Dynamically constructs detailed, dialect-aware prompts for query optimization.
    - **Response Parsing**: Validates and parses the structured JSON returned by the LLM.
    - **RAG Logic**: Implements the Retrieval-Augmented Generation for the chatbot by building a context document from the schema, query, and analysis results.

### Frontend Architecture

The frontend is a modern web application built with Next.js and the App Router.

- **Component-Based**: The UI is broken down into reusable React components (e.g., `ProblemCard`, `Chatbot`, `SandboxModal`), primarily located in the `app/dashboard/components/` directory.
- **State Management**: State is managed locally within components or shared through React Context/props.
- **Type Safety**: TypeScript is used throughout the frontend to ensure robust, error-free code.
- **Styling**: Utility-first styling is handled by Tailwind CSS, with pre-built, accessible components from shadcn/ui.

## 📂 Project Structure

```
.
├── pg-audit-backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application and endpoints
│   │   ├── models.py             # Pydantic data models
│   │   ├── db_inspector.py       # Database interaction logic (Factory Pattern)
│   │   └── ai_analyzer.py        # OpenAI API interaction logic
│   ├── .env                      # Environment variables
│   └── requirements.txt          # Python dependencies
│
└── pg-audit-frontend/
    ├── app/
    │   ├── page.tsx              # Landing/Login page (DbUriForm)
    │   └── dashboard/
    │       ├── page.tsx          # Main dashboard page component
    │       └── components/
    │           ├── AnalysisOptions.tsx
    │           ├── Chatbot.tsx
    │           ├── ProblemCard.tsx
    │           ├── SandboxModal.tsx
    │           └── ... (other UI components)
    ├── .env.local                # Frontend environment variables
    ├── next.config.mjs           # Next.js configuration
    └── package.json              # NPM dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A running PostgreSQL or MySQL database.
- An **OpenAI API Key**.

### 1\. Backend Setup

```bash
# Clone the repository
git clone https://github.com/Sumitsh28/pg-audit-backend
cd pg-audit-backend

# Install dependencies
pip install -r requirements.txt

```

**`.env` file:**

```ini
OPENAI_API_KEY="sk-..."
```

**Run Command:**
```ini
python run.py
```

The API will be available at `http://127.0.0.1:8000` or `http://localhost:8000`.

### 2\. Frontend Setup

```bash
# Clone the repository
git clone https://github.com/Sumitsh28/pg-audit-frontend
cd pg-audit-frontend

# Install dependencies
npm install

```

**`.env.local` file:**

```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Run Command:**
```ini
npm run dev
```

The Frontend will be available at `http://localhost:3000`
 in your browser.

## 🔌 API Endpoints

All endpoints are relative to the base URL (e.g., `http://127.0.0.1:8000`).

| Endpoint                       | Method | Description                                                                                                                               |
| :----------------------------- | :----- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `/start-analysis-session`      | `POST` | Initiates an analysis. Takes a `db_uri` and a `mode` (`auto`, `benchmark`, or `file`). Returns a list of identified `Problem` objects.    |
| `/get-optimization-suggestion` | `POST` | Requests an AI-powered optimization for a single query. Returns an `OptimizationResult` containing suggestions and performance estimates. |
| `/apply`                       | `POST` | Executes a DDL statement (e.g., a suggested `CREATE INDEX`) against the database. Requires a write-enabled `write_db_uri`.                |
| `/chat-on-query`               | `POST` | Sends a user's question to the RAG-based chatbot for a specific query context. Returns the AI's `ChatResponse`.                           |
| `/verify-queries`              | `POST` | Executes the original and optimized queries in a sandbox and compares their results for correctness. Returns a `SandboxResult`.           |

---



## 📸 Screenshots
![Logo](https://github.com/Sumitsh28/images/blob/main/Screenshot%202025-09-14%20at%207.34.13%E2%80%AFPM%201.png?raw=true)

![Logo](https://github.com/Sumitsh28/images/blob/main/Screenshot%202025-09-14%20at%207.34.34%E2%80%AFPM.png?raw=true)

![Logo](https://github.com/Sumitsh28/images/blob/main/Screenshot%202025-09-14%20at%207.37.28%E2%80%AFPM.png?raw=true)

![Logo](https://github.com/Sumitsh28/images/blob/main/Screenshot%202025-09-14%20at%207.41.11%E2%80%AFPM.png?raw=true)











## Demo
![Watch Here!](https://drive.google.com/file/d/10oYOsLu_0BOSoJET_PCkCdawq9mXqZtF/view?usp=drive_link)
