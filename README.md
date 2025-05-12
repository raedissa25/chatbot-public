Ce chatbot intelligent a été développé dans le cadre de mon PFE pour offrir une solution d’analyse préliminaire d’ECG accessible au grand public. Il permet à tout utilisateur de :

📤 Uploader une image d’ECG (formats : JPG, JPEG, PNG)

🤖 Analyser automatiquement l’image à l’aide d’un modèle CNN (CNN_model1.h5) pour détecter des anomalies (Normal, MI, HMI et AHB)

🧾 Générer un rapport médical structuré en langage naturel via l’API Gemini

💬 Dialoguer avec une IA médicale intégrée pour poser des questions liées à la santé cardiaque

📄 Télécharger un rapport Word (.docx) prêt à être imprimé ou partagé avec un professionnel de santé

📁 Fichiers utilisés dans ce chatbot :

app.py : Script principal de l’application Streamlit

resnet50v2_ecg_best_model.h5 : sauvegardé dans le dossier modéles, un Modèle Resnet50v2 de classification ECG par image

.env : À créer manuellement. Ce fichier doit être placé dans le même dossier que app.py et CNN_model1.h5, et doit contenir votre clé API Gemini sous la forme :

GEMINI_API_KEY="ton GEMINI API"

requirements.txt : Dépendances nécessaires à l'exécution de l'application

⚠️ Important : Ce chatbot ne remplace pas un diagnostic médical. Il fournit une analyse préliminaire à titre informatif uniquement.

Réalisé par : Raed Ben Aissa

Encadré par : LT COL Mohamed Hachemi Jeridi et DR Mouna Azaiz
