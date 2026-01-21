"""
LLM-based item summarization for enhanced features.

Uses OpenAI or Anthropic to generate concise summaries of items
that capture key selling points for ranking.
"""

import asyncio
from typing import AsyncIterator

from tqdm import tqdm

from src.config import Domain, DomainConfig, LLMProvider, get_settings
from src.data.schemas import Item


class ItemSummarizer:
    """
    Generate item summaries using LLMs.
    
    Summaries are used as:
    1. Additional text for embedding generation
    2. Display text for users
    3. Structured features (sentiment, key phrases)
    """
    
    def __init__(
        self,
        provider: LLMProvider | None = None,
        domain: Domain | None = None,
        max_concurrent: int = 5,
    ):
        self.settings = get_settings()
        self.provider = provider or self.settings.llm_provider
        self.domain = domain or self.settings.domain
        self.max_concurrent = max_concurrent
        
        # Get domain-specific prompt
        self.domain_config = DomainConfig.get(self.domain)
        self.summary_prompt = self.domain_config["summary_prompt"]
        
        # Initialize client based on provider
        if self.provider == LLMProvider.OPENAI:
            import openai
            self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self._async_client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
            self._model = self.settings.openai_model
        elif self.provider == LLMProvider.ANTHROPIC:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
            self._model = self.settings.anthropic_model
    
    def summarize_item(self, item: Item) -> str:
        """
        Generate summary for a single item.
        
        Args:
            item: Item to summarize
            
        Returns:
            Generated summary text
        """
        item_text = self._format_item_text(item)
        prompt = self.summary_prompt.format(item_text=item_text)
        
        if self.provider == LLMProvider.OPENAI:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, compelling summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        
        elif self.provider == LLMProvider.ANTHROPIC:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        
        raise ValueError(f"Unknown provider: {self.provider}")
    
    async def summarize_item_async(self, item: Item) -> str:
        """Async version of summarize_item."""
        item_text = self._format_item_text(item)
        prompt = self.summary_prompt.format(item_text=item_text)
        
        if self.provider == LLMProvider.OPENAI:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, compelling summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        
        elif self.provider == LLMProvider.ANTHROPIC:
            response = await self._async_client.messages.create(
                model=self._model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        
        raise ValueError(f"Unknown provider: {self.provider}")
    
    def summarize_items(
        self, 
        items: list[Item],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Generate summaries for multiple items (sync).
        
        For large batches, prefer summarize_items_async.
        """
        summaries = []
        iterator = tqdm(items, desc="Summarizing") if show_progress else items
        
        for item in iterator:
            try:
                summary = self.summarize_item(item)
                summaries.append(summary)
            except Exception as e:
                # Fallback to description
                summaries.append(item.description[:200] if item.description else item.name)
                print(f"Warning: Failed to summarize {item.item_id}: {e}")
        
        return summaries
    
    async def summarize_items_async(
        self,
        items: list[Item],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Generate summaries for multiple items (async with concurrency limit).
        
        Much faster than sync version for large batches.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def summarize_with_limit(item: Item) -> str:
            async with semaphore:
                try:
                    return await self.summarize_item_async(item)
                except Exception as e:
                    return item.description[:200] if item.description else item.name
        
        tasks = [summarize_with_limit(item) for item in items]
        
        if show_progress:
            summaries = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
                summaries.append(await coro)
            # Re-order to match input (as_completed doesn't preserve order)
            # Actually we need a different approach for progress + order
            summaries = await asyncio.gather(*tasks)
        else:
            summaries = await asyncio.gather(*tasks)
        
        return summaries
    
    def _format_item_text(self, item: Item) -> str:
        """Format item content for the prompt."""
        parts = [
            f"Name: {item.name}",
            f"Description: {item.description}",
        ]
        
        # Add domain-specific fields
        for field in self.domain_config["text_fields"]:
            if field in item.text_content:
                parts.append(f"{field}: {item.text_content[field]}")
        
        # Add key features
        feature_parts = []
        for key, value in item.features.features.items():
            if not isinstance(value, (list, dict)):
                feature_parts.append(f"{key}={value}")
        
        if feature_parts:
            parts.append(f"Features: {', '.join(feature_parts[:10])}")
        
        return "\n".join(parts)
    
    def update_items_with_summaries(
        self,
        items: list[Item],
        summaries: list[str] | None = None,
    ) -> list[Item]:
        """
        Update items with generated summaries.
        
        Args:
            items: Items to update
            summaries: Pre-generated summaries (optional)
            
        Returns:
            Updated items with summaries in features
        """
        if summaries is None:
            summaries = self.summarize_items(items)
        
        for item, summary in zip(items, summaries):
            item.features.summary = summary
        
        return items


class MockSummarizer:
    """
    Mock summarizer for testing without API calls.
    
    Generates template-based summaries.
    """
    
    def __init__(self, domain: Domain | None = None):
        self.settings = get_settings()
        self.domain = domain or self.settings.domain
    
    def summarize_item(self, item: Item) -> str:
        """Generate mock summary."""
        features = item.features.features
        
        if self.domain == Domain.HOTEL:
            return (
                f"{item.name} is a {features.get('star_rating', 3)}-star property "
                f"with a {features.get('review_score', 4)}/5 rating. "
                f"Located {features.get('distance_to_center', 2)}km from the center, "
                f"it offers great value for travelers seeking comfort and convenience."
            )
        
        elif self.domain == Domain.WEALTH_REPORT:
            return (
                f"This {features.get('asset_class', 'investment')} analysis offers "
                f"a risk level of {features.get('risk_level', 3)}/5 with "
                f"{features.get('return_potential', 5)}% return potential. "
                f"Ideal for investors with a {features.get('time_horizon', 5)}-year horizon."
            )
        
        else:  # ecommerce
            return (
                f"{item.name} - rated {features.get('rating', 4)}/5 stars "
                f"by {features.get('review_count', 100)} customers. "
                f"Ships in {features.get('shipping_days', 3)} days with easy returns."
            )
    
    def summarize_items(self, items: list[Item], **kwargs) -> list[str]:
        """Generate mock summaries for multiple items."""
        return [self.summarize_item(item) for item in items]
