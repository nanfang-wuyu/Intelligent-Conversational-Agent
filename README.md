# Intelligent Conversational Agent

This repository contains the project for **"Advanced Topics in Artificial Intelligence (ATAI)"**, focusing on building an intelligent conversational agent that can handle multiple types of questions using various datasets. (The project is tested on the **Speakeasy** web-based infrastructure, provided by the course teaching group.)

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Types of Questions](#types-of-questions)
- [Project Structure](#project-structure)

## Project Overview

The goal of this project is to design and implement a conversational agent capable of answering questions based on diverse datasets. The project is structured around answering five different types of questions and relies on various AI and NLP techniques to provide intelligent responses.

The main logic for the conversational agent is implemented in `main.ipynb`. It is able to run with local temps and was evaluated on the Speakeasy platform.

## Datasets

The project utilizes the following datasets:
1. **Knowledge Graph about Movies** – Used to answer factual and recommendation questions.
2. **Multimedia Dataset** – Needed for answering multimedia questions.
3. **Pre-trained Knowledge Graph Embeddings** – Used to handle embedding-based questions.
4. **Crowdsourcing Dataset** – Essential for answering crowdsourcing questions.

Datasets may be uploaded in future. All datasets are pre-processed and cleaned using according scripts provided in root directory.

## Types of Questions

The conversational agent is tested on the following types of questions:
1. **Factual Questions** – Answered using the knowledge graph about movies.
2. **Embedding Questions** – Answered using pre-trained knowledge graph embeddings.
3. **Multimedia Questions** – Multimedia data is used to provide accurate answers.
4. **Recommendation Questions** – Answered based on the knowledge graph.
5. **Crowdsourcing Questions** – The crowdsourcing dataset is used for these answers.

## Project Structure

- `/Information/` - Contains the detailed project description.
- `/Learning Materials/` - Some learning materials useful for agent development.
- `/datasets/` - Ignored in GitHub, may be uploaded in the future.
- `main.ipynb` - Main notebook implementing the chatbot.
- `*.ipynb` - Python scripts for cleaning, handling, and organizing data.
- `README.md` - This file.

