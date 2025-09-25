import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
import sqlite3
import pandas as pd
import openai
import os
import time
import datetime
import logging
from tqdm import tqdm
from collections import defaultdict

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# === CONFIGURATION ===
LANGS = [
    ("DE", "german"),
    ("ES", "spanish"),
    ("FR", "french"),
    ("IT", "italian"),
    ("PO", "polish"),
    ("PT", "portuguese"),
    ("RU", "russian"),
    ("SV", "swedish"),
    ("TR", "turkish"),
]
ORIG_DB_TEMPLATE = "./lang_variation/ID_babylonGPTKB_seed1_{lang}_temp0.db"
NEW_DB_TEMPLATE = "./lang_variation/backtranslation_EN/babylonGPTKB_{lang}-EN_{timestamp}.db"
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4.1-mini-2025-04-14"
batch_size = 20
sleep_time = 3

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === RETRY WRAPPER ===
def with_retries(func, *args, max_retries=1000000, initial_wait=1, **kwargs):
    attempt = 0
    wait = initial_wait
    while attempt < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(
                f"[!] DB operation failed (attempt {attempt+1}/{max_retries}): {e}"
            )
            time.sleep(wait)
            wait *= 2
            attempt += 1
    # after all attempts failed
    raise sqlite3.OperationalError(f"Operation failed after {max_retries} retries.")

# === COLLECT UNIQUE STRINGS ===
def collect_unique_texts():
    texts = set()
    node = with_retries(pd.read_sql, "SELECT name, first_parent FROM node", conn)
    predicate = with_retries(pd.read_sql, "SELECT name FROM predicate", conn)
    for col in ['name', 'first_parent']:
        texts.update(node[col].dropna().astype(str).unique())
    texts.update(predicate['name'].dropna().astype(str).unique())

    return sorted(texts)

# === TRANSLATE WITH OPENAI ===
def translate_texts_openai(texts, lang_name):
    translated = {}
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating {lang_name.title()}..."):
        chunk = texts[i:i+batch_size]
        joined_text = "\n".join(chunk)
        try:
            client = openai.Client()
            prompt = (
                f"You are translating a list of {lang_name} node and predicate labels from a knowledge graph into English. "
                f"For each {lang_name} term below, return only the English translation on a separate line. "
                f"DO NOT omit or skip any terms. Preserve the order of the list. {lang_name.title()} terms:\n{joined_text}"
            )
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=0
            )
            result = response.output_text
            translations = result.strip().split('\n')
            if len(translations) != len(chunk):
                logging.warning(f"[!] Mismatch in translation count: expected {len(chunk)}, got {len(translations)}")
            for original, translation in zip(chunk, translations):
                translated[original] = translation.strip()
        except Exception as e:
            logging.warning(f"[!] OpenAI error: {e}")
            for text in chunk:
                translated[text] = text
        time.sleep(sleep_time)
    return translated

# === MAKE TRANSLATIONS UNIQUE ===
def make_translations_unique(translation_map):
    reverse_map = defaultdict(list)
    for original, translated in translation_map.items():
        reverse_map[translated].append(original)

    new_map = {}
    for translated, originals in reverse_map.items():
        for i, original in enumerate(originals):
            if i == 0:
                new_map[original] = translated
            else:
                new_map[original] = f"{translated} ({i})"
    return new_map

# === UPDATE TABLE WITH PROGRESS BAR ===
def update_table(table_name, columns, translation_map):
    df = with_retries(pd.read_sql, f"SELECT * FROM {table_name}", conn)

    if "id" not in df.columns:
        logging.warning(f"No 'id' column found in table '{table_name}'. Skipping.")
        return

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: translation_map.get(x, x))

    df['__original_id'] = df['id']

    if table_name in ['node', 'predicate']:
        unique_col = 'name'
        duplicate_mask = df.duplicated(subset=[unique_col], keep=False)
        if duplicate_mask.any():
            skipped_ids = df.loc[duplicate_mask, '__original_id'].tolist()
            logging.warning(f"{len(skipped_ids)} rows in column '{unique_col}' of table '{table_name}' left unchanged during translation.")
            df = df[~duplicate_mask]

    cursor = conn.cursor()
    updated_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Updating {table_name}"):
        set_cols = [col for col in columns if col in row]
        # For triple table, ensure uniqueness after translation
        if table_name == "triple" and "object" in set_cols:
            orig_object = row["object"]
            translated_object = translation_map.get(str(orig_object), str(orig_object))
            subject = row["subject"] if "subject" in row else None
            predicate = row["predicate"] if "predicate" in row else None
            # Check for uniqueness
            unique_object = translated_object
            suffix = 1
            while True:
                cursor.execute(
                    "SELECT id FROM triple WHERE subject = ? AND predicate = ? AND object = ? AND id != ?",
                    (subject, predicate, unique_object, row["id"])
                )
                if cursor.fetchone() is None:
                    break
                unique_object = f"{translated_object} ({suffix})"
                suffix += 1
            # Update the row with the unique object
            update_values = []
            update_cols = []
            for col in set_cols:
                if col == "object":
                    update_values.append(unique_object)
                else:
                    update_values.append(row[col])
                update_cols.append(col)
            placeholders = ", ".join([f"{col} = ?" for col in update_cols])
            update_values.append(row["__original_id"])
            def execute_update():
                cursor.execute(f"UPDATE {table_name} SET {placeholders} WHERE id = ?", update_values)
            try:
                with_retries(execute_update)
                updated_count += 1
            except sqlite3.IntegrityError as e:
                logging.warning(f"[!] Skipping id {row['id']} due to integrity error: {e}")
            except sqlite3.InterfaceError as e:
                logging.warning(f"[!] Skipping id {row['id']} due to interface error: {e}")
        else:
            values = [row[col] for col in set_cols]
            placeholders = ", ".join([f"{col} = ?" for col in set_cols])
            values.append(row["__original_id"])
            def execute_update():
                cursor.execute(f"UPDATE {table_name} SET {placeholders} WHERE id = ?", values)
            try:
                with_retries(execute_update)
                updated_count += 1
            except sqlite3.IntegrityError as e:
                logging.warning(f"[!] Skipping id {row['id']} due to integrity error: {e}")
            except sqlite3.InterfaceError as e:
                logging.warning(f"[!] Skipping id {row['id']} due to interface error: {e}")

    with_retries(conn.commit)
    logging.info(f"Updated {updated_count} rows in '{table_name}' table.")

# === MAIN WORKFLOW ===
for lang, lang_name in LANGS:
    original_db_path = ORIG_DB_TEMPLATE.format(lang=lang)
    new_db_path = NEW_DB_TEMPLATE.format(lang=lang, timestamp=timestamp)
    logging.info(f"\n=== Processing {lang_name.title()} ({lang}) ===")
    os.system(f"cp {original_db_path} {new_db_path}")
    conn = sqlite3.connect(new_db_path)

    logging.info("Collecting unique texts...")
    all_texts = collect_unique_texts()

    logging.info(f"Translating texts via OpenAI for {lang_name.title()}...")
    translation_map = translate_texts_openai(all_texts, lang_name)
    translation_map = make_translations_unique(translation_map)

    logging.info("Applying translations to database...")
    update_table("node", ["name", "first_parent"], translation_map)
    update_table("predicate", ["name"], translation_map)
    update_table("triple", ["object"], translation_map)

    with_retries(conn.close)
    logging.info(f"\n✅ Translation complete. New database saved to:\n{new_db_path}")
