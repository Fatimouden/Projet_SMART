from fastapi import FastAPI
import pandas as pd
app=FastAPI()

@app.get("/forecast")
def get_forecast(city: str = "Nice"):
    url = f"https://www.meteociel.fr/previsions-arome-1h/2005/{city.lower()}.htm"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    # Ici : parser le HTML avec BeautifulSoup (si possible)
    # ou utiliser Selenium pour charger JS

    # Pour cet exemple, on renvoie une réponse fictive :
    return {"city": city, "temperature": 24}
