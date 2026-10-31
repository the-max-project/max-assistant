# Copyright (c) 2025-2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for finding people, understanding relationships,
and querying the user's family, friend, and support networks.
"""
from enum import Enum
import json
import logging
from typing import Annotated, Any, Dict

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from max_assistant.clients.neo4j_client import (
    Neo4jCircuitBreakerError,
    Neo4jClient,
    Neo4jClientError,
)
from max_assistant.models.location_models import LocationDetails
from max_assistant.models.person_models import PersonDetails
from max_assistant.tools.registry import BaseToolProvider
from max_assistant.utils.decorators import requires_db

logger = logging.getLogger(__name__)


class RelationshipCategory(str, Enum):
    """Categorical kinship, social, and care relationship types."""
    # Direct Lineage & Kinship
    DESCENDANT = "descendant"
    ANCESTOR = "ancestor"
    SIBLINGS = "siblings"
    AUNT_UNCLE = "aunt_uncle"
    SPOUSE = "spouse"

    # In-Law Relationships
    SIBLING_IN_LAW = "sibling_in_law"
    CHILD_IN_LAW = "child_in_law"
    IN_LAW = "in_law"

    # Social & Support Network
    FRIEND = "friend"
    SUPPORT_WORKER = "support_worker"
    COHABITANT = "cohabitant"


class PersonTools(BaseToolProvider):
    """
    Encapsulates person-related tools including profile retrieval, contact searches,
    relationship role resolutions, and graph network queries relative to the :User node.
    """

    MIN_DEPTH = 1
    MAX_DEPTH = 5

    MAP_SIGNATURE_TO_RELATION = {
        # --- 1 Hop Relationships ---
        (('MARRIED_TO', True),): {"F": "wife", "M": "husband", "default": "spouse"},
        (('MARRIED_TO', False),): {"F": "wife", "M": "husband", "default": "spouse"},
        (('PARTNER_OF', True),): {"F": "partner", "M": "partner", "default": "partner"},
        (('PARTNER_OF', False),): {"F": "partner", "M": "partner", "default": "partner"},
        (('PARENT_OF', True),): {"F": "daughter", "M": "son", "default": "child"},
        (('PARENT_OF', False),): {"F": "mother", "M": "father", "default": "parent"},
        (('FRIEND_OF', True),): {"default": "friend"},
        (('FRIEND_OF', False),): {"default": "friend"},
        (('SUPPORTED_BY', True),): {"default": "support worker"},
        (('LIVES_WITH', True),): {"default": "cohabitant"},
        (('LIVES_WITH', False),): {"default": "cohabitant"},

        # --- 2 Hop Relationships ---
        # Sibling: Path goes up to parent (incoming), then down to sibling (outgoing)
        (('PARENT_OF', False), ('PARENT_OF', True)): {
            "F": "sister",
            "M": "brother",
            "default": "sibling",
        },
        # Grandchild: Path goes down to child (outgoing), then down to grandchild (outgoing)
        (('PARENT_OF', True), ('PARENT_OF', True)): {
            "F": "granddaughter",
            "M": "grandson",
            "default": "grandchild",
        },
        # Grandparent: Path goes up to parent (incoming), then up to grandparent (incoming)
        (('PARENT_OF', False), ('PARENT_OF', False)): {
            "F": "grandmother",
            "M": "grandfather",
            "default": "grandparent",
        },
    }

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama | None = None):
        """Initializes the toolset with a specific Neo4j client."""
        super().__init__(db_client, llm)
        logger.debug("PersonTools initialized with a Neo4j client.")

    # -------------------------------------------------------------------------
    # Relationship Description & Path Resolvers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_relationship_description(path_data: Dict[str, Any], user_id: str) -> str:
        """Convert a Neo4j path into a human-readable description."""
        rel_details = path_data.get('rel_details', [])
        gender = path_data.get('gender') or "unknown"

        if not rel_details:
            return "related"

        path_signature = []
        current_node = user_id

        for rel in rel_details:
            is_outgoing = (rel['start'] == current_node)
            path_signature.append((rel['type'], is_outgoing))
            current_node = rel['end'] if is_outgoing else rel['start']

        path_tuple = tuple(path_signature)
        role_mapping = PersonTools.MAP_SIGNATURE_TO_RELATION.get(path_tuple)
        if role_mapping:
            return role_mapping.get(gender, role_mapping.get("default", "related"))

        return f"connected ({rel_details[0]['type']})"

    async def _find_relationship_path(self, person_id: str, user_id: str) -> Dict[str, Any] | None:
        """Finds the shortest relationship path from :User to a person by person ID."""
        params = {"person_id": person_id, "user_id": user_id}

        family_query = """
            MATCH path = shortestPath((u:User {id: $user_id})-[r:MARRIED_TO|PARENT_OF|PARTNER_OF*1..5]-(p {id: $person_id}))
            WHERE u <> p
            RETURN [r IN relationships(path) | {
                type: type(r),
                start: startNode(r).id,
                end: endNode(r).id
            }] AS rel_details, p.gender AS gender
            ORDER BY length(path) ASC
            LIMIT 1
        """
        try:
            result = await self.db_client.execute_query(family_query, params)
            if result.get("data"):
                logger.debug("Found family path for id=%s", person_id)
                return result["data"][0]

            other_query = """
                MATCH path = shortestPath((u:User {id: $user_id})-[*1..5]-(p {id: $person_id}))
                WHERE u <> p
                RETURN [r IN relationships(path) | {
                    type: type(r),
                    start: startNode(r).id,
                    end: endNode(r).id
                }] AS rel_details, p.gender AS gender
                ORDER BY length(path) ASC
                LIMIT 1
            """
            result = await self.db_client.execute_query(other_query, params)
            if result.get("data"):
                logger.debug("Found other path for id=%s", person_id)
                return result["data"][0]

            return None

        except Neo4jClientError as e:
            logger.warning("Database error while searching for relationship path: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Traversal Builder for query_relationship
    # -------------------------------------------------------------------------

    def _build_match_clause(self, category: RelationshipCategory, depth: int) -> str:
        """Constructs targeted Cypher traversals per relationship type."""
        match category:
            case RelationshipCategory.DESCENDANT:
                return f"""
                    MATCH (u:User {{id: $user_id}})-[:PARENT_OF*{depth}]->(target:Person)
                """

            case RelationshipCategory.ANCESTOR:
                return f"""
                    MATCH (u:User {{id: $user_id}})<-[:PARENT_OF*{depth}]-(target:Person)
                """

            case RelationshipCategory.SIBLINGS:
                return """
                    MATCH (u:User {id: $user_id})<-[:PARENT_OF]-(parent:Person)-[:PARENT_OF]->(target:Person)
                    WHERE target.id <> u.id
                """

            case RelationshipCategory.AUNT_UNCLE:
                if depth == 1:
                    return """
                        MATCH (u:User {id: $user_id})<-[:PARENT_OF]-(parent:Person)
                        MATCH (parent)<-[:PARENT_OF]-(grandparent:Person)-[:PARENT_OF]->(target:Person)
                        WHERE target.id <> parent.id
                          AND NOT (u)<-[:PARENT_OF]-(target)
                    """
                ancestor_hops = depth - 1
                return f"""
                    MATCH (u:User {{id: $user_id}})<-[:PARENT_OF*{ancestor_hops}]-(lineal:Person)
                    MATCH (lineal)<-[:PARENT_OF]-(g_parent:Person)-[:PARENT_OF]->(target:Person)
                    WHERE target.id <> lineal.id
                      AND NOT (u)<-[:PARENT_OF *1..{depth}]-(target)
                """

            case RelationshipCategory.SPOUSE:
                return """
                    MATCH (u:User {id: $user_id})-[:MARRIED_TO]-(target:Person)
                    WHERE target.id <> u.id
                """

            case RelationshipCategory.CHILD_IN_LAW:
                return f"""
                    MATCH (u:User {{id: $user_id}})-[:PARENT_OF*{depth}]->(descendant:Person)-[:MARRIED_TO]-(target:Person)
                    WHERE target.id <> u.id
                """

            case RelationshipCategory.IN_LAW:
                return f"""
                    MATCH (u:User {{id: $user_id}})-[:MARRIED_TO]-(spouse:Person)
                    MATCH (spouse)<-[:PARENT_OF*{depth}]-(target:Person)
                    WHERE target.id <> u.id AND target.id <> spouse.id
                """

            case RelationshipCategory.SIBLING_IN_LAW:
                return """
                    MATCH (u:User {id: $user_id})
                    CALL {
                        // Path A: Spouses of user's siblings
                        WITH u
                        MATCH (u)<-[:PARENT_OF]-(:Person)-[:PARENT_OF]->(sibling:Person)-[:MARRIED_TO]-(target:Person)
                        WHERE sibling.id <> u.id AND target.id <> u.id
                        RETURN target
                        UNION
                        // Path B: Siblings of user's spouse
                        WITH u
                        MATCH (u)-[:MARRIED_TO]-(spouse:Person)<-[:PARENT_OF]-(:Person)-[:PARENT_OF]->(target:Person)
                        WHERE spouse.id <> u.id AND target.id <> spouse.id AND target.id <> u.id
                        RETURN target
                    }
                """

            case RelationshipCategory.FRIEND:
                return """
                    MATCH (u:User {id: $user_id})-[:FRIEND_OF]-(target:Person)
                    WHERE target.id <> u.id
                """

            case RelationshipCategory.SUPPORT_WORKER:
                return """
                    MATCH (u:User {id: $user_id})-[:SUPPORTED_BY]->(target:Person)
                """

            case RelationshipCategory.COHABITANT:
                return """
                    MATCH (u:User {id: $user_id})-[:LIVES_WITH]-(target:Person)
                    WHERE target.id <> u.id
                """

    def _compose_cypher(self, category: RelationshipCategory, depth: int, include_spouses: bool) -> str:
        """Assembles match patterns and projection stages."""
        traversal = self._build_match_clause(category, depth).strip()

        # Non-genealogical queries or direct marriage queries do not expand spouses
        skip_spouse_expansion = (
            not include_spouses
            or category in (
                RelationshipCategory.SPOUSE,
                RelationshipCategory.CHILD_IN_LAW,
                RelationshipCategory.SIBLING_IN_LAW,
                RelationshipCategory.FRIEND,
                RelationshipCategory.SUPPORT_WORKER,
                RelationshipCategory.COHABITANT,
            )
        )

        if skip_spouse_expansion:
            return f"""
                {traversal}
                RETURN DISTINCT properties(target) AS person
            """

        # Return both the relative and their spouse (excluding current user)
        return f"""
            {traversal}
            OPTIONAL MATCH (target)-[:MARRIED_TO]-(spouse:Person)
            WHERE spouse.id <> $user_id
            WITH target, spouse
            UNWIND [target] + (CASE WHEN spouse IS NOT NULL THEN [spouse] ELSE [] END) AS member
            RETURN DISTINCT properties(member) AS person
        """

    # -------------------------------------------------------------------------
    # Public Tools
    # -------------------------------------------------------------------------

    @requires_db
    async def query_relationship(
        self,
        user_info: Annotated[dict, InjectedState("userinfo")],
        relationship: str,
        depth: int = 1,
        include_spouses: bool = False,
    ) -> str:
        """Queries the graph database for people connected to the user by relationship type, including
        family connections, ancestors, descendants, friends and support staff/professionals.

        Args:
            relationship: The category of connection. Must be one of:
                - 'descendant': Children, grandchildren, great-grandchildren
                - 'ancestor': Parents, grandparents, great-grandparents
                - 'siblings': Brothers and sisters
                - 'sibling_in_law': Brothers-in-law, sisters-in-law (spouse's siblings or siblings' spouses)
                - 'child_in_law': Sons-in-law, daughters-in-law (spouse of child/grandchild)
                - 'in_law': Parents-in-law (parents/grandparents of spouse)
                - 'aunt_uncle': Aunts, uncles, grandaunts, granduncles
                - 'spouse': Current husband, wife, or partner
                - 'friend': Personal friends (FRIEND_OF)
                - 'support_worker': Caregivers, support workers, medical aides (SUPPORTED_BY)
                - 'cohabitant': Roommates or individuals sharing the residence (LIVES_WITH)
            depth: Generational distance modifier (1-5, default is 1).
                   Applies to: 'descendant', 'ancestor', 'aunt_uncle', 'child_in_law', and 'in_law'.
                   Rule: Standard (parent, child, aunt, child-in-law) = 1.
                         'grand' variations (grandparent, grandchild, grandaunt) = 2.
                         Add +1 for each additional 'great-' prefix encountered.
                   Ignored for 'siblings', 'sibling_in_law', 'spouse', 'friend', 'support_worker', and 'cohabitant'.
            include_spouses: If True, returns both the relatives and their spouses (default False).
        """
        logger.info("Tool: query_relationship (category=%s, depth=%s, include_spouses=%s)", relationship, depth, include_spouses)
        user_id = self._get_verified_user_id(user_info)

        if not (isinstance(depth, int) and self.MIN_DEPTH <= depth <= self.MAX_DEPTH):
            return json.dumps({
                "error": "Invalid_Depth",
                "details": f"Depth parameter must be an integer between {self.MIN_DEPTH} and {self.MAX_DEPTH}."
            })

        try:
            category = RelationshipCategory(relationship.strip().lower())
        except ValueError:
            valid_types = ", ".join(k.value for k in RelationshipCategory)
            return json.dumps({
                "error": "Invalid_Relationship",
                "details": f"Relationship '{relationship}' is not recognized. Valid options: {valid_types}."
            })

        query = self._compose_cypher(category, depth, include_spouses)
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    @requires_db
    async def find_person_by_name(
        self,
        user_info: Annotated[dict, InjectedState("userinfo")],
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> str:
        """Finds a person, family member, friend, or support contact by first name, last name, or both.
        Returns potential matches with all attributes (phone, email, notes, address) and a relationship
        description field. At least one name must be provided. Case-insensitive.
        """
        logger.info("Tool: find_person_by_name: fn=%s, ln=%s", first_name, last_name)
        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[*1..5]-(p:Person)
            WHERE ($first_name IS NULL OR toLower(p.firstName) CONTAINS $first_name)
              AND ($last_name IS NULL OR toLower(p.lastName) CONTAINS $last_name)
            RETURN properties(p) AS person, labels(p) AS labels
            LIMIT 10
        """
        params = {
            "first_name": first_name.lower() if first_name else None,
            "last_name": last_name.lower() if last_name else None,
            "user_id": user_id,
        }

        try:
            result = await self.db_client.execute_query(query, params)

            validated_results = []
            for item in result.get("data", []):
                person_props = item.get("person")
                person_labels = item.get("labels")

                if person_props:
                    validated_person = PersonDetails.model_validate(person_props)
                    person_id = validated_person.id

                    path_data = await self._find_relationship_path(person_id, user_id)
                    relationship_desc = (
                        self._get_relationship_description(path_data, user_id)
                        if path_data else "unknown"
                    )

                    validated_results.append({
                        "person": validated_person.model_dump(mode="json"),
                        "labels": person_labels,
                        "relationship": relationship_desc,
                    })

            return json.dumps(validated_results, indent=2, default=str)

        except Neo4jCircuitBreakerError as e:
            logger.warning("Circuit Breaker blocked find_person_by_name query: %s", e)
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The database is currently offline. Inform the user you cannot access their data right now.",
                "details": str(e),
            })
        except Neo4jClientError as e:
            logger.error("Database error in find_person_by_name: %s", e)
            return json.dumps({"error": "Database_Unavailable", "message": str(e)})
        except ValidationError as e:
            logger.error("Validation error for PersonDetails: %s", e.errors())
            return json.dumps({"error": "Data validation failed", "details": e.errors()}, default=str)
        except Exception as e:
            logger.error("Unexpected error in find_person_by_name: %s", e)
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    @requires_db
    async def find_person_by_title(
        self,
        title: str,
        user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """Finds a person by title or profession (e.g. 'Doctor', 'Nurse', 'Therapist').
        Searches the 'title' field across Person and Support nodes for a case-insensitive match.
        """
        logger.info("Tool: find_person_by_title: title=%s", title)
        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[*1..2]-(p:Person)
            WHERE toLower(p.title) CONTAINS $title
            RETURN properties(p) AS person
            LIMIT 10
        """
        params = {"title": title.lower(), "user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    @requires_db
    async def get_relationship_to_user(
        self,
        first_name: str,
        last_name: str,
        user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """Finds the relationship between the user and a specific person identified by first and last name."""
        user_id = self._get_verified_user_id(user_info)

        find_query = """
            MATCH (u:User {id: $user_id})-[*1..3]-(p:Person)
            WHERE toLower(p.firstName) = $first_name AND toLower(p.lastName) = $last_name
            RETURN p.id AS person_id
            LIMIT 1
        """
        params = {
            "first_name": first_name.lower(),
            "last_name": last_name.lower(),
            "user_id": user_id,
        }

        try:
            find_result = await self.db_client.execute_query(find_query, params)
            if not find_result.get("data"):
                return json.dumps({"error": "Person not found", "details": "No person found with that name."})

            person_id = find_result["data"][0].get("person_id")
            if not person_id:
                return json.dumps({"error": "Data parsing failed", "details": "Person found, but has no 'id' property."})

            path_data = await self._find_relationship_path(person_id, user_id)
            description = (
                self._get_relationship_description(path_data, user_id)
                if path_data else "unknown"
            )

            return json.dumps({
                "relationship": description,
                "path_length": len(path_data.get("rel_details", [])) if path_data else 0,
            }, indent=2)

        except Neo4jCircuitBreakerError as e:
            logger.warning("Circuit Breaker blocked get_relationship_to_user: %s", e)
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The database is offline. Inform the user you cannot access their data right now.",
                "details": str(e),
            })
        except Neo4jClientError as e:
            logger.error("Database error in get_relationship_to_user: %s", e)
            return json.dumps({"error": "Database_Unavailable", "message": str(e)})
        except Exception as e:
            logger.error("Unexpected error in get_relationship_to_user: %s", e)
            return json.dumps({"error": "Internal_Error", "details": str(e)})

    @requires_db
    async def get_user_info_internal(self, username: str) -> Dict[str, Any]:
        """Internal helper to fetch user and residence information as a dictionary."""
        logger.info("Internal: get_user_info_internal for %s", username)

        query = """
            MATCH (u:User {userName: $username})
            OPTIONAL MATCH (u)-[:LIVES_AT]->(l:Location)
            RETURN properties(u) AS user, properties(l) AS location
            LIMIT 1
        """
        params = {"username": username}

        try:
            result = await self.db_client.execute_query(query, params)
            if not result.get("data"):
                return {"error": "User not found.", "details": f"No User node found with userName = {username}"}

            data = result["data"][0]
            user_props = data.get("user")
            location_props = data.get("location")

            if not user_props:
                return {"error": "Data parsing failed", "details": "Found a user record but no properties."}

            validated_user = PersonDetails.model_validate(user_props)
            validated_location = LocationDetails.model_validate(location_props) if location_props else None

            return {
                "user": validated_user.model_dump(mode="json"),
                "location": validated_location.model_dump(mode="json") if validated_location else None,
            }

        except Neo4jCircuitBreakerError as e:
            logger.warning("Circuit Breaker blocked get_user_info_internal: %s", e)
            return {"error": "Database_Unavailable", "details": str(e)}
        except Neo4jClientError as e:
            logger.error("Database error in get_user_info_internal: %s", e)
            return {"error": "Database_Unavailable", "details": str(e)}
        except ValidationError as e:
            logger.error("Validation error for User/Location: %s", e.errors())
            return {"error": "User record failed validation.", "details": e.errors()}
        except Exception as e:
            logger.error("Unexpected error in get_user_info_internal: %s", e)
            return {"error": "Data parsing failed", "details": str(e)}

    # -------------------------------------------------------------------------
    # Tool Registration
    # -------------------------------------------------------------------------

    def get_tools(self) -> list[StructuredTool]:
        """Returns all person, contact, and relationship tools bound to this provider."""
        return [
            StructuredTool.from_function(
                name="query_relationship",
                func=None,
                coroutine=self.query_relationship,
                description=self.query_relationship.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                name="find_person_by_name",
                func=None,
                coroutine=self.find_person_by_name,
                description=self.find_person_by_name.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                name="find_person_by_title",
                func=None,
                coroutine=self.find_person_by_title,
                description=self.find_person_by_title.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                name="get_relationship_to_user",
                func=None,
                coroutine=self.get_relationship_to_user,
                description=self.get_relationship_to_user.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]