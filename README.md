# travel-multi-agent-workshop

## Deployment Instructions
To deploy this solution to your Azure account, follow these steps:
1. **Clone the Repository**: Start by cloning this repository to your local machine.
    ```bash
    git clone https://github.com/Azure-Samples/travel-multi-agent-workshop.git
    cd travel-multi-agent-workshop
    ``` 
2. **Install Prerequisites**: Ensure you have the following installed:
   - [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
    - [Python 3.8+](https://www.python.org/downloads/)
    - [Node.js and npm](https://nodejs.org/en/download/)
3. **Login to Azure**: Use the Azure CLI to log in to your Azure account.
    ```bash
    az login
    ```
4. **Run azd up**: Navigate to the `travel-multi-agent-workshop/02_completed/infra` directory and run the following command to deploy the solution:
    ```bash
    azd up
    ```
   This command will provision all necessary Azure resources and seed the database. It may take several minutes to complete.

## Setting up local development

When you deploy this solution, it automatically configures `.env` files with the required Azure endpoints and authentication tokens for both the main application and MCP server.

To run the solution locally after deployment:

### Terminal 1 - Start the MCP Server:

Open a new terminal, navigate to the `02_completed` directory, then run:

**Linux/macOS:**
```bash
source venv/bin/activate
cd mcp_server
PYTHONPATH=../python python mcp_http_server.py
```

**Windows (PowerShell):**
.\venv\Scripts\Activate.ps1
cd mcp_server
$env:PYTHONPATH="..\python"; python mcp_http_server.py
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
set PYTHONPATH=../python && python mcp_http_server.py
```

### Terminal 2 - Start the Travel API:

Open a new terminal, navigate to the `02_completed` directory, then run:

**Linux/macOS:**
```bash
source venv/bin/activate
cd python
uvicorn src.app.travel_agents_api:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
cd python
uvicorn src.app.travel_agents_api:app --reload --host 0.0.0.0 --port 8000
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
uvicorn src.app.travel_agents_api:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 3 - Start the Frontend:

Open a new terminal, navigate to the `02_completed\frontend` folder, then run:

**All platforms:**
```bash
npm install
npm start
```

Access the applications:

- Travel API: [http://localhost:8000/docs](http://localhost:8000/docs)
- MCP Server: [http://localhost:8080/docs](http://localhost:8080/docs)
- Frontend: [http://localhost:4200](http://localhost:4200/)