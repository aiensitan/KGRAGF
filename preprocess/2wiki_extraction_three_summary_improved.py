import os
import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm
from llama_index.llms.ollama import Ollama


def extract_high_quality_triplets(llm, ctx):
    """Use high-quality prompts to extract triplets with strict filtering."""
    query = (
        "Extract triplets informative from the text following the examples. "
        "Make sure the triplet texts are only directly from the given text! "
        "Complete directly and strictly following the instructions without any additional words, line break nor space!\n"
        f"{'-' * 20}\n"
        "Text: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\n"
        "Triplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$"
        "<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$"
        "<Scott Derrickson##occupation##producer>$$\n"
        f"{'-' * 20}\n"
        "Text: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. "
        "It stars Shirley Temple in her final starring role as well as her final film appearance. "
        "Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\n"
        "Triplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n"
        f"{'-' * 20}\n"
        f"Text: {ctx}\n"
        "Triplets:"
    )

    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split("$$")
    ctx_lower = ctx.lower()

    for triplet_text in triplet_texts:
        triplet_text = triplet_text.replace("There are the triplets extracted from the text:\n\n", "")
        triplet_text = triplet_text.replace("There are the extracted triplets:\n\n", "")
        triplet_text = triplet_text.replace("Here are the triplets extracted from the text:\n\n", "")
        triplet_text = triplet_text.replace("Here are the extracted triplets:\n\n", "")
        triplet_text = triplet_text.replace("Triplets:", "")
        triplet_text = triplet_text.replace("Text:", "")
        triplet_text = triplet_text.strip()

        if len(triplet_text) <= 6:
            continue

        if triplet_text.startswith("<"):
            triplet_text = triplet_text[1:]
        if triplet_text.endswith(">"):
            triplet_text = triplet_text[:-1]

        tokens = triplet_text.split("##")
        if len(tokens) != 3:
            continue

        h, r, t = [tok.strip().replace("<", "").replace(">", "").strip() for tok in tokens]

        if ("no " in h.lower()) or ("no " in t.lower()) or \
                ("unknown" in h.lower()) or ("unknown" in t.lower()) or \
                ("null" in h.lower()) or ("null" in t.lower()) or \
                (len(h) < 2) or (len(t) < 2):
            continue

        meaningless_words = {"it", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i"}
        if h.lower() in meaningless_words or t.lower() in meaningless_words:
            continue

        if not (h.startswith('"') and h.endswith('"')):
            h = h.strip('"\'')
        if not (r.startswith('"') and r.endswith('"')):
            r = r.strip('"\'')
        if not (t.startswith('"') and t.endswith('"')):
            t = t.strip('"\'')

        if not any(word.lower() in ctx_lower for word in r.split()):
            if not any(word.lower() in ctx_lower for word in t.split()):
                continue

        if h.lower() == t.lower() and len(r.split()) < 2:
            continue

        if len(h) > 100 or len(t) > 100 or len(r) > 50:
            continue

        if h.lower() in t.lower() and len(h) > 5:
            continue
        if t.lower() in h.lower() and len(t) > 5:
            continue

        triplets.add((h, r, t))

    return [[h, r, t] for (h, r, t) in triplets]


def generate_entity_summary(llm, entity_name, sentences):
    context = "\n".join([f"{i}: {sent}" for i, sent in enumerate(sentences)])
    query = (
        f"Based on the following sentences about {entity_name}, generate a comprehensive summary (2-3 sentences) "
        "that captures the key information:\n\n"
        f"{context}\n\n"
        f"Please provide a concise summary of {entity_name}:"
    )
    try:
        resp = llm.complete(query)
        summary = resp.text.strip()
        if summary.startswith("Summary:"):
            summary = summary[8:].strip()
        if summary.startswith(f"{entity_name}"):
            return summary
        return f"{entity_name} {summary}" if len(summary) > 10 else None
    except Exception:
        return None


def extract_entity_relations(llm, context_entities):
    if len(context_entities) < 2:
        return []

    entity_names = list(context_entities.keys())
    context_text = ""
    for entity, sentences in context_entities.items():
        context_text += f"\n{entity}:\n"
        for i, sentence in enumerate(sentences):
            context_text += f"  {i + 1}. {sentence}\n"

    query = f"""Based on the following context, identify relationships between different entities and generate triplets in the format <Entity1##relationship##Entity2>.

Context:
{context_text}

Available entities: {', '.join(entity_names)}

Instructions:
1. Look for ANY meaningful relationships between these entities (teammates, competitors, contemporaries, from same organization, worked together, etc.)
2. The relationship can be any real connection - don't limit to specific types
3. Generate triplets in the exact format: <Entity1##relationship##Entity2>$$<Entity1##relationship##Entity2>$$...
4. Only use entities from the available entities list
5. Only include relationships that are clearly supported by the context
6. Make relationships natural and descriptive

Generate the triplets based on the context above:"""

    try:
        resp = llm.complete(query)
        triplet_text = resp.text.strip()
        return parse_entity_relations_triplets(triplet_text, entity_names)
    except Exception as e:
        print(f"Error extracting entity relations: {e}")
        return []


def parse_entity_relations_triplets(triplet_text, valid_entities):
    triplets = []
    triplet_text = triplet_text.replace("There are the triplets extracted from the text:\n\n", "")
    triplet_text = triplet_text.replace("There are the extracted triplets:\n\n", "")
    triplet_text = triplet_text.replace("Triplets:", "").strip()

    if "$$" in triplet_text:
        triplet_candidates = triplet_text.split("$$")
    else:
        triplet_candidates = []
        for line in triplet_text.split("\n"):
            if "<" in line and "##" in line and ">" in line:
                triplet_candidates.append(line)

    for candidate in triplet_candidates:
        candidate = candidate.strip()
        if len(candidate) <= 6:
            continue
        if candidate.startswith("<"):
            candidate = candidate[1:]
        if candidate.endswith(">"):
            candidate = candidate[:-1]

        tokens = candidate.split("##")
        if len(tokens) != 3:
            continue

        h, r, t = [tok.strip().replace("<", "").replace(">", "").strip() for tok in tokens]

        h_valid = False
        t_valid = False
        for valid_entity in valid_entities:
            if (h.lower() == valid_entity.lower() or
                    valid_entity.lower() in h.lower() or
                    h.lower() in valid_entity.lower()):
                h = valid_entity
                h_valid = True
            if (t.lower() == valid_entity.lower() or
                    valid_entity.lower() in t.lower() or
                    t.lower() in valid_entity.lower()):
                t = valid_entity
                t_valid = True

        if not (h_valid and t_valid):
            continue
        if h.lower() == t.lower():
            continue
        if len(r) < 2 or r.lower() in ["no", "unknown", "null", "the", "and", "or"]:
            continue
        if not (r.startswith('"') and r.endswith('"')):
            r = r.strip('"\'')
        if len(h) > 100 or len(r) > 100 or len(t) > 100:
            continue
        if len(h) > 1 and len(r) > 1 and len(t) > 1:
            triplets.append([h, r, t])

    return triplets


def extract_entity_sentence_relations(llm, entity_name, sentences, light_llm=None):
    relations = {}
    if light_llm is None:
        light_llm = Ollama(model="phi", device="cuda", request_timeout=60)

    for i, sentence in enumerate(sentences):
        analysis_query = f"""Analyze the relationship between the entity and this sentence. What type of information does this sentence provide about the entity?

Entity: {entity_name}
Sentence: {sentence}

Question: What aspect or type of information about {entity_name} is described in this sentence?
Answer with a brief phrase (e.g., "biographical information", "career details", "educational background", "achievements", "personal life", etc.):"""

        try:
            analysis_resp = llm.complete(analysis_query)
            analysis_text = analysis_resp.text.strip()

            triplet_query = f"""Generate a triplet showing the relationship between the entity and the sentence content. Follow the exact format:

Entity: {entity_name}
Sentence: {sentence}
Relationship type: {analysis_text}

Create ONE triplet in this exact format:
<{entity_name}##[relationship]##[content_type]>

Where:
- [relationship] should be "describes", "contains", "mentions", or "includes"
- [content_type] should be the type of information (like "career information", "biographical details", "educational background")

Examples:
<Harold Miner##describes##career achievements>
<University of Southern California##contains##educational information>
<NBA Slam Dunk Contest##mentions##competition details>

Generate the triplet:"""

            triplet_resp = light_llm.complete(triplet_query)
            triplet_text = triplet_resp.text.strip()
            triplets = parse_entity_sentence_triplets(triplet_text, entity_name)
            if triplets:
                relations[str(i)] = triplets
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            continue

    return relations


def parse_entity_sentence_triplets(triplet_text, entity_name):
    triplets = []
    triplet_text = triplet_text.replace("There are the triplets extracted from the text:\n\n", "")
    triplet_text = triplet_text.replace("There are the extracted triplets:\n\n", "")
    triplet_text = triplet_text.replace("Triplet:", "").strip()

    if "$$" in triplet_text:
        triplet_candidates = triplet_text.split("$$")
    else:
        triplet_candidates = [triplet_text]

    for candidate in triplet_candidates:
        candidate = candidate.strip()
        if len(candidate) <= 6:
            continue
        if candidate.startswith("<"):
            candidate = candidate[1:]
        if candidate.endswith(">"):
            candidate = candidate[:-1]

        tokens = candidate.split("##")
        if len(tokens) != 3:
            continue

        h, r, t = [tok.strip().replace("<", "").replace(">", "").strip() for tok in tokens]

        if not (h.lower() == entity_name.lower() or
                entity_name.lower() in h.lower() or
                h.lower() in entity_name.lower()):
            h = entity_name

        if len(r) < 2 or r.lower() in ["no", "unknown", "null", "the", "and", "or"]:
            continue
        if len(t) < 2 or t.lower() in ["no", "unknown", "null", "the", "and", "or"]:
            continue

        if not (t.startswith('"') and t.endswith('"')):
            t = t.strip('"\'')

        valid_relations = [
            "describes", "contains", "mentions", "includes", "shows", "presents",
            "details", "covers", "discusses", "highlights", "explains", "provides",
            "reveals", "indicates", "states"
        ]
        if not any(rel in r.lower() for rel in valid_relations):
            if len(r.split()) > 3:
                continue

        if (len(h) > 1 and len(r) > 1 and len(t) > 1 and
                len(h) < 100 and len(r) < 50 and len(t) < 100):
            triplets.append([h, r, t])

    return triplets


def _parse_context(sample):
    ctxs = sample.get("context")
    if isinstance(ctxs, str):
        try:
            ctxs = json.loads(ctxs)
        except Exception:
            return []
    return ctxs or []


def process_batch_improved(batch_data):
    process_id = os.getpid()
    batch_id = batch_data[0].get("_id", "unknown") if batch_data else "unknown"
    llm = Ollama(model="llama3:8b", device="cuda", request_timeout=120)
    light_llm = Ollama(model="phi", device="cuda", request_timeout=60)

    batch_entities = set()
    processed_entities = 0

    for sample in batch_data:
        ctxs = _parse_context(sample)
        for ctx in ctxs:
            ent = ctx[0]
            batch_entities.add(ent)

    total_entities = len(batch_entities)
    print(f"\n[Process {process_id}] Starting batch {batch_id} with {total_entities} unique entities")

    all_entity_summaries = {}
    for sample in batch_data:
        ctxs = _parse_context(sample)
        for ctx in ctxs:
            ent = ctx[0]
            out_path = os.path.join(out_dir, f"{ent.replace('/', '_')}.json")
            if os.path.exists(out_path):
                processed_entities += 1
                print(f"[Process {process_id}] Entity {ent} already processed ({processed_entities}/{total_entities})")
                continue

            sentences = ctx[1]
            t_summary_start = time.perf_counter()
            entity_summary = generate_entity_summary(llm, ent, sentences)
            t_summary_end = time.perf_counter()
            if entity_summary:
                all_entity_summaries[ent] = entity_summary
            print(f"[Process {process_id}] Entity {ent} summary time: {t_summary_end - t_summary_start:.2f}s")

    for sample in batch_data:
        ctxs = _parse_context(sample)

        sample_context_entities = {}
        for sample_ctx in ctxs:
            sample_ent = sample_ctx[0]
            sample_sentences = sample_ctx[1]
            sample_context_entities[sample_ent] = sample_sentences
        t_entity_rel_start = time.perf_counter()
        all_entity_relations = extract_entity_relations(llm, sample_context_entities)
        t_entity_rel_end = time.perf_counter()
        print(f"[Process {process_id}] Sample {batch_id} entity-rel time: {t_entity_rel_end - t_entity_rel_start:.2f}s")

        for ctx in ctxs:
            ent = ctx[0]
            out_path = os.path.join(out_dir, f"{ent.replace('/', '_')}.json")
            if os.path.exists(out_path):
                continue
            t_entity_start = time.perf_counter()

            entity_kg = {
                "entity_summary": None,
                "entity_sentence_relations": {},
                "sentence_relations": {},
                "relations": [],
                "original_sentences": ctx[1],
            }

            sentences = ctx[1]
            if ent in all_entity_summaries:
                entity_kg["entity_summary"] = all_entity_summaries[ent]

            t_entity_sent_start = time.perf_counter()
            entity_sentence_rels = extract_entity_sentence_relations(
                llm, ent, sentences, light_llm=light_llm
            )
            t_entity_sent_end = time.perf_counter()
            if entity_sentence_rels:
                entity_kg["entity_sentence_relations"] = entity_sentence_rels
            print(
                f"[Process {process_id}] Entity {ent} entity-sentence time: "
                f"{t_entity_sent_end - t_entity_sent_start:.2f}s"
            )

            sentence_relations = {}
            for i, sentence in enumerate(sentences):
                ctx_text = sentence if i == 0 else f"{ent}: {sentence}"
                t_sentence_rel_start = time.perf_counter()
                triplets = extract_high_quality_triplets(llm, ctx_text)
                t_sentence_rel_end = time.perf_counter()
                if triplets:
                    sentence_relations[i] = triplets
                print(
                    f"[Process {process_id}] Entity {ent} sentence {i} triplets time: "
                    f"{t_sentence_rel_end - t_sentence_rel_start:.2f}s"
                )

            if sentence_relations:
                entity_kg["sentence_relations"] = sentence_relations

            entity_specific_relations = []
            if all_entity_relations:
                for relation in all_entity_relations:
                    if len(relation) == 3:
                        h, r, t = relation
                        if (h.lower() == ent.lower() or ent.lower() in h.lower() or h.lower() in ent.lower() or
                                t.lower() == ent.lower() or ent.lower() in t.lower() or t.lower() in ent.lower()):
                            entity_specific_relations.append(relation)

            if entity_specific_relations:
                entity_kg["relations"] = entity_specific_relations

            if any([
                entity_kg["entity_summary"],
                entity_kg["relations"],
                entity_kg["entity_sentence_relations"],
                entity_kg["sentence_relations"],
            ]):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(entity_kg, f, ensure_ascii=False, indent=2)
                processed_entities += 1

                t_entity_end = time.perf_counter()
                print(
                    f"[Process {process_id}] Processed entity {ent} "
                    f"({processed_entities}/{total_entities}), entity total time: "
                    f"{t_entity_end - t_entity_start:.2f}s"
                )
            else:
                processed_entities += 1
                t_entity_end = time.perf_counter()
                print(
                    f"[Process {process_id}] No useful information extracted for entity {ent} "
                    f"({processed_entities}/{total_entities}), entity total time: "
                    f"{t_entity_end - t_entity_start:.2f}s"
                )

    print(f"[Process {process_id}] Batch {batch_id} completed. Processed {processed_entities} entities")
    return processed_entities


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "2wikimultihopQA" / "train_200.jsonl"
    with data_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    global out_dir
    out_dir = Path(__file__).resolve().parents[2] / "data" / "2wikimultihopQA" / "kgs" / "extract_subkgs_train_200_llama3_three_relations_improved"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir)

    num_processes = min(cpu_count(), 2)
    batch_size = 10

    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])

    print("\n=== 2WikiMultihopQA Three-Relation Extraction ===")
    print(f"Total samples: {len(data)}")
    print(f"Number of batches: {len(batches)}")
    print(f"Processes: {num_processes}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {out_dir}")
    print("===============================================\n")

    with Pool(num_processes) as pool:
        processed_counts = list(
            tqdm(pool.imap(process_batch_improved, batches), total=len(batches), desc="Processing batches")
        )

    print("\n=== Extraction Summary ===")
    print(f"Total batches processed: {len(batches)}")
    print(f"Total entities processed: {sum(processed_counts)}")
    print("===================================")
