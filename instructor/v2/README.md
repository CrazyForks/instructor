# Instructor V2 Migration

This directory contains the new architecture for the instructor library, implementing a clean adapter pattern for better maintainability and provider support.

## Architecture Overview

The v2 architecture introduces:

- **InstructorClient**: Main client class with retry logic and template rendering
- **ProviderAdapter**: Abstract base class defining the adapter interface  
- **Provider-specific adapters**: Concrete implementations for each LLM provider
- **from_provider()**: Factory function to create clients from provider strings

### Key Features

- Jinja2 template rendering for message content with context
- Usage tracking with a custom Usage class
- Retry logic with tenacity
- Support for both structured (with response_model) and unstructured (response_model=None) completions
- Clean separation of concerns between providers

## Current Migration Status

Currently, **only OpenAI functionality has been migrated** with the following limitations:
- ✅ Sync completions only
- ❌ No streaming support
- ❌ No async support
- ❌ Only basic modes supported

## Provider and Mode Migration Status

Based on analysis of `instructor/process_response.py` and the existing v2 implementation, here's the comprehensive migration status:

| Provider | Mode | Streaming | Parallel Tools | Async | Sync | Is Migrated |
|----------|------|-----------|----------------|--------|------|-------------|
| **OpenAI** | TOOLS | ❌ | ❌ | ❌ | ✅ | 🟡 Partial |
| OpenAI | TOOLS_STRICT | ❌ | ❌ | ❌ | ✅ | 🟡 Partial |
| OpenAI | JSON | ❌ | ❌ | ❌ | ✅ | 🟡 Partial |
| OpenAI | JSON_SCHEMA | ❌ | ❌ | ❌ | ✅ | 🟡 Partial |
| OpenAI | PARALLEL_TOOLS | ❌ | ✅ | ❌ | ✅ | 🟡 Partial |
| OpenAI | JSON_O1 | ❌ | ❌ | ❌ | ❌ | ❌ No |
| OpenAI | MD_JSON | ❌ | ❌ | ❌ | ❌ | ❌ No |
| OpenAI | FUNCTIONS | ❌ | ❌ | ❌ | ❌ | ❌ No (Deprecated) |
| OpenAI | RESPONSES_TOOLS | ❌ | ❌ | ❌ | ❌ | ❌ No |
| OpenAI | RESPONSES_TOOLS_WITH_INBUILT_TOOLS | ❌ | ❌ | ❌ | ❌ | ❌ No |
| **Anthropic** | ANTHROPIC_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Anthropic | ANTHROPIC_REASONING_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Anthropic | ANTHROPIC_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Anthropic | ANTHROPIC_PARALLEL_TOOLS | ❌ | ✅ | ✅ | ✅ | ❌ No |
| **Google/Vertex** | VERTEXAI_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Google/Vertex | VERTEXAI_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Google/Vertex | VERTEXAI_PARALLEL_TOOLS | ❌ | ✅ | ✅ | ✅ | ❌ No |
| Google/Vertex | GEMINI_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Google/Vertex | GEMINI_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Google/Vertex | GENAI_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Google/Vertex | GENAI_STRUCTURED_OUTPUTS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Mistral** | MISTRAL_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Mistral | MISTRAL_STRUCTURED_OUTPUTS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Cohere** | COHERE_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Cohere | COHERE_JSON_SCHEMA | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Cerebras** | CEREBRAS_TOOLS | ❌ | ❌ | ✅ | ✅ | ❌ No |
| Cerebras | CEREBRAS_JSON | ❌ | ❌ | ✅ | ✅ | ❌ No |
| **Fireworks** | FIREWORKS_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Fireworks | FIREWORKS_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Writer** | WRITER_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Writer | WRITER_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Bedrock** | BEDROCK_TOOLS | ✅ | ❌ | ✅ | ✅ | ❌ No |
| Bedrock | BEDROCK_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **Perplexity** | PERPLEXITY_JSON | ✅ | ❌ | ✅ | ✅ | ❌ No |
| **OpenRouter** | OPENROUTER_STRUCTURED_OUTPUTS | ✅ | ❌ | ✅ | ✅ | ❌ No |

### Legend
- ✅ = Supported in v1
- ❌ = Not supported in v1 or not migrated
- 🟡 = Partially migrated
- 🟢 = Fully migrated

## Migration Priority

Based on usage patterns and provider importance, suggested migration priority:

### High Priority (Core providers)
1. **OpenAI** - Complete streaming, async, and missing modes
2. **Anthropic** - Full migration with streaming and parallel tools
3. **Google/Vertex AI** - Complete ecosystem migration

### Medium Priority (Popular providers)  
4. **Mistral** - Growing ecosystem
5. **Cohere** - Enterprise usage
6. **Fireworks** - Performance-focused

### Lower Priority (Specialized providers)
7. **Cerebras** - Specialized hardware
8. **Writer** - Niche use case
9. **Bedrock** - AWS ecosystem
10. **Perplexity** - Search-focused
11. **OpenRouter** - Proxy service

## Implementation Guidelines

Each provider adapter should implement:

```python
class ProviderAdapter(ABC):
    id: str  # e.g. "anthropic"
    modes: set[Mode]  # supported modes
    
    def build_request(self, response_model, mode, messages, hooks, **kwargs) -> dict:
        """Build provider-specific request"""
        
    def process_response(self, response, response_model, mode, strict, context, hooks) -> BaseModel:
        """Process provider-specific response"""
        
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming"""
        
    def supports_parallel_tools(self) -> bool:
        """Whether provider supports parallel tool calling"""
```

## Key Migration Considerations

1. **Streaming Support**: Most providers support streaming, requiring async generators
2. **Parallel Tools**: Only some providers support multiple tool calls per request
3. **Message Formats**: Each provider has unique message formatting requirements
4. **Error Handling**: Provider-specific error types and retry logic
5. **Usage Tracking**: Different usage metrics across providers

## Testing Strategy

Each migrated provider should have:
- Sync and async completion tests
- Streaming tests (where supported)
- Parallel tools tests (where supported)
- Error handling and retry tests
- Usage tracking tests

## Current V2 Structure

```
instructor/v2/
├── __init__.py           # Public API exports
├── client.py             # InstructorClient with retry logic
├── auto_client.py        # from_provider() factory function
└── adapters/
    ├── __init__.py       # Adapter registry
    ├── base.py           # ProviderAdapter base class
    ├── openai.py         # OpenAI adapter (partial)
    └── registry.py       # Adapter registration system
```

## Next Steps

1. Complete OpenAI adapter with streaming and async support
2. Implement Anthropic adapter
3. Add comprehensive test suite for each provider
4. Document migration guide for other contributors
5. Gradually migrate remaining providers based on priority 