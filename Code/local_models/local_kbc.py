import openai as oa
from openai import OpenAI
import json
import queue
import time
import threading
import logging

# === CONFIGURATION ===
max_iterations = 500000000  # no limits
verbose = True
nthreads = 50

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("kbc_extraction.log"),
        logging.StreamHandler()
    ]
)

# === LLM WRAPPER ===
def promptLLMLocal(message):
    client = OpenAI(base_url="https://llm.scads.ai/v1", api_key='insert_your_api_key_here')
    r = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model=
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        #"meta-llama/Llama-3.3-70B-Instruct",
        #"openai/gpt-oss-120b",
        #"openGPT-X/Teuken-7B-instruct-research-v0.4",
        #"deepseek-ai/DeepSeek-R1"

        temperature=0
    )
    return r.choices[0].message.content

# === FILE UTILS ===
def append_to_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data) + '\n')

def remove_json_delimiters(input_string):
    if input_string.startswith("```json"):
        input_string = input_string[len("```json"):].strip()
    if input_string.startswith("```"):
        input_string = input_string[len("```"):].strip()
    if input_string.startswith("\n\n```json"):
        input_string = input_string[len("\n\n```json"):].strip()
    if input_string.startswith("\n\n```json\n"):
        input_string = input_string[len("\n\n```json\n"):].strip()
    if input_string.startswith("\n```json"):
        input_string = input_string[len("\n```json"):].strip()
    if input_string.startswith(" ```json"):
        input_string = input_string[len(" ```json"):].strip()
    if input_string.startswith("  ```json"):
        input_string = input_string[len("  ```json"):].strip()
    if input_string.endswith("```"):
        input_string = input_string[:-len("```")].strip()
    return input_string

def readFirstNSubjects(n):
    return loadSubjectQueue()[:n]

def deleteFirstNsubjects(n):
    subject_queue = loadSubjectQueue()
    with open(subject_queue_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(subject_queue[n:]))

# === PATHS ===
base_path = "./"
subject_queue_path = base_path + "subjectQueue.json" # initially, a list containing the seed entity
processed_subjects_path = base_path + "processedSubjects.json" # initially, an empty list
triples_output_path = base_path + "triples.jsonl"
parse_errors_path = base_path + "batchResultParseErrors.jsonl"
not_ne_path = base_path + "notNE.jsonl"

# === MAIN FUNCTION: TRIPLE EXTRACTION ===
def getTriples(s, result_queue, errorQueue):
    prompt = (
        'I want to construct a knowledge graph on the topic of the ancient city of Babylon. '
        'Given a subject entity, return all facts that you know for the subject as a list of '
        'subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, '
        'for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.\n\n'
        'Important: \n- If you don\'t know the subject, return an empty list. \n- If the subject is not a named entity, return an empty list.\n'
        '- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.\n'
        '- If the subject is a named entity, include at least one triple where predicate is "instanceOf".\n'
        '- Do not get too wordy.\n- Separate several objects into multiple triples with one object.\n'
        '- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.\n'
        '- Don\'t generate any other text except for the triples in JSON format.\n'
        f'Subject: "{s}"'
    )
    logging.info(f"Querying subject: {s}")
    try:
        output_string = promptLLMLocal(prompt)
        try:
            linetriples = json.loads(remove_json_delimiters(output_string))
            result_queue.put(linetriples)
            logging.info(f"   Received {len(linetriples)} triples for: {s}")
        except Exception as e:
            append_to_jsonl_file(parse_errors_path, {"error": str(e), "line": str(output_string)})
            result_queue.put([])
            logging.warning(f"   Failed to parse JSON for: {s}")
    except Exception as e:
        errorQueue.put(e)
        logging.exception(f"Exception during prompt for subject: {s}")

# === DEDUPLICATION ===
def deduplicate_triples(triples):
    seen = set()
    unique_triples = []
    for t in triples:
        try:
            key = (t.get('s', '').strip(), t.get('p', '').strip(), t.get('o', '').strip())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)
        except Exception as e:
            logging.error(f"Error processing triple {t}: {e}")
            continue
    return unique_triples

def storeTriples(newTriples):
    unique_triples = deduplicate_triples(newTriples)
    logging.info(f"   Deduplicated {len(newTriples) - len(unique_triples)} triples (from {len(newTriples)} to {len(unique_triples)})")
    with open(triples_output_path, 'a', encoding='utf8') as file:
        for obj in unique_triples:
            file.write(json.dumps(obj) + '\n')

# === SUBJECT QUEUE MGMT ===
def loadProcessedSubjects():
    try:
        with open(processed_subjects_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except:
        return []

def loadSubjectQueue():
    try:
        with open(subject_queue_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except:
        return []

def appendToProcessedSubjects(subjects):
    processedSubjects = loadProcessedSubjects()
    processedSubjects.extend(subjects)
    with open(processed_subjects_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(processedSubjects))

def appendToSubjectQueue(newSubjects):
    subject_queue = loadSubjectQueue()
    subject_queue += newSubjects
    with open(subject_queue_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(subject_queue))

# === ENTITY LITERAL CHECK ===
def IsLiteral(o):
    response = promptLLMLocal(
        f'I want you to perform named entity recognition (NER) on the topic of the ancient city of Babylon. '
        f'Your task is to classify if a given phrase is a topic-relevant named entity, or not.\n\n'
        f'- If the phrase is a named entity, return "false".\n'
        f'- If the phrase is not a named entity, return "true".\n'
        f'- Return only true or false.\n'
        f'Phrase: "{o}"'
    )
    result = response.strip().lower()
    if result == 'true':
        append_to_jsonl_file(not_ne_path, {'text': o})
        return True
    return result != 'false'

def getTotalNodeCount():
    processed = set(loadProcessedSubjects())
    queued = set(loadSubjectQueue())
    
    try:
        with open(triples_output_path, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    triple = json.loads(line)
                    processed.add(triple.get("s", ""))
                    processed.add(triple.get("o", ""))
                except:
                    continue
    except FileNotFoundError:
        pass
    
    return len(processed.union(queued))


# === MAIN LOOP ===
logging.info("==== Knowledge Extraction Pipeline Started ====")

for i in range(max_iterations):
    newSubjects = readFirstNSubjects(nthreads)

    if len(newSubjects) == 0:
        logging.info("No more subjects in queue. Exiting loop.")
        break

    threads = []
    result_queue = queue.Queue()
    errorQueue = queue.Queue()

    for subj in newSubjects:
        thread = threading.Thread(target=getTriples, args=(subj, result_queue, errorQueue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if errorQueue.empty():
        allTriples = []
        while not result_queue.empty():
            allTriples.extend(result_queue.get())

        storeTriples(allTriples)

        newSubjectsFromObjects = []
        allObjects = [t.get('o', '') for t in allTriples]
        processedSubjects = loadProcessedSubjects()
        subject_queue = loadSubjectQueue()

        for o in allObjects:
            if o in processedSubjects or o in newSubjectsFromObjects or o in subject_queue:
                continue
            if IsLiteral(o):
                continue
            newSubjectsFromObjects.append(o)

        appendToProcessedSubjects(newSubjects)

        deleteFirstNsubjects(nthreads)

        total_nodes = getTotalNodeCount()
        #limit = 300
        #if total_nodes > limit:
        #    logging.warning(f"Node count exceeded limit (current: {total_nodes}, limit: {limit}). Stopping process.")
        #    break

        appendToSubjectQueue(newSubjectsFromObjects)

        if len(newSubjectsFromObjects) == 0 and len(loadSubjectQueue()) == 0:
            logging.info("No new subjects found and queue is now empty. Exiting.")
            break

        logging.info(f"Iteration {i} completed.")
        logging.info(f"Queue size: {len(loadSubjectQueue())}, Processed: {len(loadProcessedSubjects())}")
        logging.info(f"{time.strftime('%X %x %Z')}\n")
        time.sleep(1)
    else:
        error_msg = str(errorQueue.get())[:100]
        logging.error(f"Error in thread processing: {error_msg}")
        time.sleep(60)

logging.info("==== Knowledge Extraction Pipeline Finished ====")
