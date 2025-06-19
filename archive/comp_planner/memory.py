import datetime
import json

class AgentMemory:
    def __init__(self):
        self.short_term = []  # List of (query, response, timestamp)
        self.extracted_entities = {}
        self.previous_recommendations = []

    def update_from_query(self, query, persona, co_client=None):
        """Extract and store relevant information from query using Cohere."""
        entity_prompt = f"""
        Extract key compensation planning entities from: "{query}"
        Return JSON with: role, level, location, company_stage, special_requirements
        """
        if co_client:
            try:
                response = co_client.generate(prompt=entity_prompt, max_tokens=250)
                entities = json.loads(response.generations[0].text)
                self.extracted_entities.update(entities)
            except Exception:
                pass

    def add_interaction(self, query, response):
        self.short_term.append({
            "query": query,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self.short_term = self.short_term[-10:]  # Keep last 10

    def store_recommendation(self, recommendation):
        self.previous_recommendations.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "recommendation": recommendation
        })
        self.previous_recommendations = self.previous_recommendations[-5:]

    def get_context(self):
        return {
            "extracted_entities": self.extracted_entities,
            "conversation_history": self.short_term,
            "previous_recommendations": self.previous_recommendations
        }