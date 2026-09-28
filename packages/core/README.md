# core

Config, clients, and helpers common to all apps: settings, chat models, embeddings, blob storage,
RabbitMQ, hashing, database sessions.

Independent of `domain` — nothing here imports a model, so the two packages stay siblings.
Provider swaps (Ollama ↔ Bedrock, MinIO ↔ S3) are config changes; new providers are new
`Embedder` subclasses.
