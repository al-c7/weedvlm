"""
This script extracts the metadata for each dataset from the Weed-AI dataset's
HTML page. The user needs to download the page itself for now.
"""

from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

def extract_dataset(html_file):
    with open(html_file, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    result = {}

    # Dataset title
    h4 = soup.find("h4")
    if h4:
        result["name"] = h4.get_text(strip=True)

    # JSON-LD metadata
    jsonld = soup.find("script", {"type": "application/ld+json"})
    if jsonld:
        data = json.loads(jsonld.string)

        result["published"] = data.get("datePublished")
        result["description"] = data.get("description")

        result["authors"] = [
            author["name"]
            for author in data.get("creator", [])
        ]

        result["citation"] = data.get("citation")
        result["license"] = data.get("license")

    # Sample image count
    sample_header = soup.find(
        string=re.compile(r"Sample of \d+ Images")
    )
    if sample_header:
        m = re.search(r"(\d+)", sample_header)
        if m:
            result["images"] = int(m.group(1))

    # Annotation statistics table
    result["annotation_statistics"] = []

    for row in soup.select("tbody tr"):
        cols = [
            c.get_text(strip=True)
            for c in row.find_all("td")
        ]

        if len(cols) == 4:
            result["annotation_statistics"].append({
                "category": cols[0],
                "images": int(cols[1]),
                "segments": int(cols[2]),
                "bounding_boxes": int(cols[3])
            })

    # Cards (Crop, Photography, Other Details)
    sections = {}

    for card in soup.select(".MuiCard-root"):
        heading = card.find("h4")
        value = card.find("p", class_=re.compile("jss33"))

        if heading and value:
            sections[
                heading.get_text(strip=True)
            ] = value.get_text(" ", strip=True)

    result["properties"] = sections

    return result


if __name__ == "__main__":
    all_datasets = []
    for html_file in Path(".dataset-pages").glob("*.html"):
        try:
            dataset = extract_dataset(html_file)
            all_datasets.append(dataset)
            print("Processed {html_file.name}")
        except Exception as e:
            print(f"Failed {html_file.name}: {e}")

    with open("all_datasets.json", "w", encoding="utf-8") as f:
        json.dump(all_datasets, f, indent=2)