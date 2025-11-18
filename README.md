--- Google Gemini API ---
```
GOOGLE_API_KEY=""
gpt_deployment_name="gemini-2.5-flash"
embed_deployment_name="models/text-embedding-004"
```

--- MS SQL Database ---
```
MS_SQL_SERVER=""
MS_SQL_DATABASE=""
MS_SQL_USER=""
MS_SQL_PASSWORD=""
MS_SQL_DRIVER="ODBC Driver 17 for SQL Server"
```
```
deactivate
Remove-Item -Recurse -Force venv 
python -m venv venv
pip install -r requirements.txt
.\venv\Scripts\Activate.ps1
or,
.\venv\Scripts\activate
```
