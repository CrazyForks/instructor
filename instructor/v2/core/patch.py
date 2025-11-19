from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Protocol, TypeVar

from typing_extensions import ParamSpec

from ...cache import BaseCache, load_cached_response, make_cache_key, store_cached_response
from ...core.hooks import Hooks
from ...core.patch import handle_context
from ...mode import Mode
from ...templating import handle_templating
from ...utils import is_async
from ...utils.providers import Provider
from .registry import mode_registry, normalize_mode
from .retry import retry_async_v2, retry_sync_v2

T_Model = TypeVar("T_Model")
T_ParamSpec = ParamSpec("T_ParamSpec")
T_Retval = TypeVar("T_Retval")


class PatchedCreateCallable(Protocol):
    def __call__(
        self,
        response_model: type[Any] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: Any = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


def _maybe_get_cache_key(
    *,
    cache: BaseCache | None,
    response_model: type[Any] | None,
    kwargs: dict[str, Any],
    mode: Mode,
) -> tuple[str | None, Any]:
    if cache is None or response_model is None:
        return None, None
    messages = (
        kwargs.get("messages")
        or kwargs.get("contents")
        or kwargs.get("chat_history")
    )
    key = make_cache_key(
        messages=messages,
        model=kwargs.get("model"),
        response_model=response_model,
        mode=mode.value if hasattr(mode, "value") else str(mode),
    )
    cached = load_cached_response(cache, key, response_model)
    return key, cached


def patch_v2(
    *,
    create: Callable[T_ParamSpec, T_Retval],
    provider: Provider,
    mode: Mode,
) -> PatchedCreateCallable:
    """Patch provider specific create functions using the v2 registry."""
    
    # Normalize mode before handler lookup
    normalized_mode = normalize_mode(provider, mode)
    handler_cls = mode_registry.get_handler_class(provider, normalized_mode)
    func_is_async = is_async(create)

    @wraps(create)  # type: ignore[arg-type]
    async def new_create_async(
        response_model: type[Any] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: Any = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> Any:
        cache: BaseCache | None = kwargs.pop("cache", None)
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl = cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        context = handle_context(context, validation_context)

        handler = handler_cls(provider=provider, mode=normalized_mode)
        response_model, new_kwargs = handler.prepare_request(
            response_model=response_model,
            **kwargs,
        )
        new_kwargs = handle_templating(new_kwargs, mode=mode, context=context)

        cache_key, cached_response = _maybe_get_cache_key(
            cache=cache,
            response_model=response_model,
            kwargs=new_kwargs,
            mode=mode,
        )
        if cached_response is not None:
            return cached_response

        result = await retry_async_v2(
            handler=handler,
            func=create,
            response_model=response_model,
            args=args,
            kwargs=new_kwargs,
            context=context,
            max_retries=max_retries,
            strict=strict,
            hooks=hooks,
        )
        if cache is not None and cache_key is not None and result is not None:
            store_cached_response(cache, cache_key, result, ttl=cache_ttl)
        return result

    @wraps(create)  # type: ignore[arg-type]
    def new_create_sync(
        response_model: type[Any] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: Any = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> Any:
        cache: BaseCache | None = kwargs.pop("cache", None)
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl = cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        context = handle_context(context, validation_context)

        handler = handler_cls(provider=provider, mode=normalized_mode)
        response_model, new_kwargs = handler.prepare_request(
            response_model=response_model,
            **kwargs,
        )
        new_kwargs = handle_templating(new_kwargs, mode=mode, context=context)

        cache_key, cached_response = _maybe_get_cache_key(
            cache=cache,
            response_model=response_model,
            kwargs=new_kwargs,
            mode=mode,
        )
        if cached_response is not None:
            return cached_response

        result = retry_sync_v2(
            handler=handler,
            func=create,
            response_model=response_model,
            args=args,
            kwargs=new_kwargs,
            context=context,
            max_retries=max_retries,
            strict=strict,
            hooks=hooks,
        )
        if cache is not None and cache_key is not None and result is not None:
            store_cached_response(cache, cache_key, result, ttl=cache_ttl)
        return result

    return new_create_async if func_is_async else new_create_sync

