# Architecture

PurpleMem combines a few small parts rather than one large abstraction.

## Storage

- SQLite stores structured memory rows and typed graph edges.
- Qdrant stores dense vectors for semantic retrieval.

## Retrieval pipeline

1. semantic retrieval from Qdrant
2. lexical fusion to sharpen rankings when the query has strong anchors
3. optional topic-aware browse when semantic hits are sparse
4. temporal penalties for superseded or historical facts

The retrieval layer is deliberately small. The point of the project is not to build a giant framework around these ideas, but to make the scoring and evaluation easy to inspect.

## Temporal validity

Facts can change. PurpleMem keeps `valid_from` and `valid_to` ranges instead of overwriting old facts. That makes current retrieval safer and historical queries possible.

## Entity graph

The graph is optional. It gives the system a lightweight structured layer for questions like:

- who works at X?
- what games does Y play?
- where does Z live?

The graph is not meant to replace retrieval. It sits next to it.

Use it when the query is naturally relational. Skip it when semantic retrieval already does the job.
