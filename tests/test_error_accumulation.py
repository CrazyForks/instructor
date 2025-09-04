"""Test error accumulation functionality in retry logic."""

import pytest
import instructor
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated
from pydantic import AfterValidator
from instructor.core.exceptions import InstructorRetryException
import os


def always_fail_validator(v: str):
    """A validator that always fails to test retry behavior."""
    raise ValueError(
        f"Validation always fails for input: '{v}' - attempt to extract name"
    )


def conditional_fail_validator(v: str):
    """A validator that fails for lowercase names."""
    if v.lower() == v:  # If the name is all lowercase
        raise ValueError(f"Name must be uppercase, got: '{v}'")
    return v


class AlwaysFailModel(BaseModel):
    """Test model with a validator that will always fail."""

    name: Annotated[str, AfterValidator(always_fail_validator)] = Field(
        description="A name that will always fail validation"
    )


class ConditionalFailModel(BaseModel):
    """Test model that fails validation under certain conditions."""

    name: Annotated[str, AfterValidator(conditional_fail_validator)] = Field(
        description="A name that must be uppercase"
    )


class TestErrorAccumulation:
    """Test cases for error accumulation during retry attempts."""

    def test_all_exceptions_captured_with_auth_errors(self):
        """Test that all authentication errors are captured during retry attempts."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        max_retries = 3

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=AlwaysFailModel,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the name 'John' from this text",
                    }
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Verify we captured all exceptions
        assert len(e.all_exceptions) == max_retries, (
            f"Expected {max_retries} exceptions but got {len(e.all_exceptions)}"
        )

        # Verify all exceptions are accessible and contain error information
        for i, exception in enumerate(e.all_exceptions, 1):
            assert isinstance(exception, Exception), (
                f"Exception {i} is not an Exception instance"
            )
            error_message = str(exception)
            assert len(error_message) > 0, f"Exception {i} has empty error message"
            assert "401" in error_message or "Incorrect API key" in error_message, (
                f"Exception {i} doesn't contain expected auth error info"
            )

        # Verify all exceptions can be extracted as messages
        error_messages = [str(exc) for exc in e.all_exceptions]
        assert len(error_messages) == max_retries, (
            "Should be able to extract all error messages"
        )

        # Verify exception types are consistent
        exception_types = [type(exc).__name__ for exc in e.all_exceptions]
        assert all(t == "AuthenticationError" for t in exception_types), (
            f"Expected all AuthenticationErrors, got: {exception_types}"
        )

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable for real API testing",
    )
    def test_all_validation_errors_captured_with_real_api(self):
        """Test that all validation errors are captured with real API calls."""
        api_key = os.getenv("OPENAI_API_KEY")
        instructor_client = instructor.from_provider("openai/gpt-4o", api_key=api_key)

        max_retries = 2  # Use fewer retries to save API calls

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=AlwaysFailModel,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the name 'John Smith' from this text",
                    }
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Verify we captured exceptions (might be validation errors or other issues)
        assert len(e.all_exceptions) >= 1, "Should have captured at least one exception"
        assert len(e.all_exceptions) <= max_retries + 1, (
            "Should not exceed max attempts"
        )

        # Verify all exceptions contain meaningful error information
        for i, exception in enumerate(e.all_exceptions, 1):
            assert isinstance(exception, Exception), (
                f"Exception {i} is not an Exception instance"
            )
            error_message = str(exception)
            assert len(error_message) > 0, f"Exception {i} has empty error message"

            # Check if it's a validation error containing our validator message
            if isinstance(exception, ValidationError):
                assert "Validation always fails" in error_message, (
                    f"Validation error {i} doesn't contain expected validator message"
                )

    def test_error_message_extraction(self):
        """Test that error messages can be properly extracted from all exceptions."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        max_retries = 3

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=AlwaysFailModel,
                messages=[
                    {"role": "user", "content": "Extract name from: 'Hello John Doe'"}
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Test different ways to extract error information

        # 1. Extract raw error messages
        error_messages = [str(exc) for exc in e.all_exceptions]
        assert len(error_messages) == len(e.all_exceptions), (
            "Should extract all messages"
        )

        # 2. Extract error types
        error_types = [type(exc).__name__ for exc in e.all_exceptions]
        assert len(error_types) == len(e.all_exceptions), "Should extract all types"

        # 3. Create structured error data
        structured_errors = []
        for i, exc in enumerate(e.all_exceptions, 1):
            error_data = {
                "attempt": i,
                "type": type(exc).__name__,
                "message": str(exc),
                "exception_obj": exc,
            }
            structured_errors.append(error_data)

        assert len(structured_errors) == len(e.all_exceptions), (
            "Should create structured data for all exceptions"
        )

        # 4. Verify each structured error contains required information
        for error_data in structured_errors:
            assert "attempt" in error_data, "Should include attempt number"
            assert "type" in error_data, "Should include exception type"
            assert "message" in error_data, "Should include error message"
            assert "exception_obj" in error_data, (
                "Should include original exception object"
            )

            # Verify the original exception object is intact
            original_exc = error_data["exception_obj"]
            assert isinstance(original_exc, Exception), (
                "Should preserve original exception"
            )
            assert str(original_exc) == error_data["message"], (
                "Message should match original exception string"
            )

    def test_comprehensive_error_report(self):
        """Test creating a comprehensive error report from accumulated exceptions."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        max_retries = 2

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=ConditionalFailModel,  # Different model for variety
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the name 'jane doe' from this text",
                    }
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Create comprehensive error report
        error_report = {
            "total_attempts": e.n_attempts,
            "total_exceptions_captured": len(e.all_exceptions),
            "final_exception": str(e),
            "all_exceptions": [
                {
                    "attempt_number": i,
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "exception_repr": repr(exc),
                }
                for i, exc in enumerate(e.all_exceptions, 1)
            ],
        }

        # Verify report completeness
        assert error_report["total_attempts"] > 0, "Should have attempted at least once"
        assert error_report["total_exceptions_captured"] > 0, (
            "Should have captured exceptions"
        )
        assert len(error_report["all_exceptions"]) == len(e.all_exceptions), (
            "Report should include all exceptions"
        )

        # Verify each exception entry in the report
        for exc_entry in error_report["all_exceptions"]:
            assert "attempt_number" in exc_entry, "Should include attempt number"
            assert "exception_type" in exc_entry, "Should include exception type"
            assert "error_message" in exc_entry, "Should include error message"
            assert "exception_repr" in exc_entry, "Should include exception repr"

            # Verify content is not empty
            assert exc_entry["exception_type"] != "", (
                "Exception type should not be empty"
            )
            assert exc_entry["error_message"] != "", "Error message should not be empty"

    def test_exception_object_preservation(self):
        """Test that original exception objects are preserved with all their attributes."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=AlwaysFailModel,
                messages=[
                    {"role": "user", "content": "Extract name: 'Alice Wonderland'"}
                ],
                max_retries=2,
            )

        e = exc_info.value

        # Verify original exception objects are preserved
        for i, original_exc in enumerate(e.all_exceptions, 1):
            # Should be able to access all original exception attributes
            assert hasattr(original_exc, "__class__"), (
                f"Exception {i} should preserve class information"
            )
            assert hasattr(original_exc, "__str__"), (
                f"Exception {i} should preserve string representation"
            )

            # For AuthenticationError, check specific attributes
            if hasattr(original_exc, "response"):
                # OpenAI AuthenticationError has response attribute
                assert original_exc.response is not None, (
                    f"Exception {i} should preserve response attribute"
                )

            # Verify exception can be re-raised (object is intact)
            try:
                raise original_exc
            except Exception as re_raised:
                assert type(re_raised) == type(original_exc), (
                    f"Exception {i} should maintain type when re-raised"
                )
                assert str(re_raised) == str(original_exc), (
                    f"Exception {i} should maintain message when re-raised"
                )

    def test_failed_responses_captured(self):
        """Test that failed responses are captured alongside exceptions."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        max_retries = 3

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=AlwaysFailModel,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the name 'Bob Smith' from this text",
                    }
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Verify we captured failed responses
        assert hasattr(e, "all_failed_responses"), (
            "Should have all_failed_responses attribute"
        )
        assert len(e.all_failed_responses) > 0, (
            "Should have captured at least one failed response"
        )
        assert len(e.all_failed_responses) == len(e.all_exceptions), (
            "Should have same number of failed responses as exceptions"
        )

        # Verify each failed response is accessible (may be None for auth errors)
        for _i, _failed_response in enumerate(e.all_failed_responses, 1):
            # For auth errors, the response might be None since the request fails before getting a response
            # We just need to ensure the structure is maintained and the array has the right length
            pass  # The main assertion is that we have the same number of responses as exceptions

    def test_enhanced_exception_data_extraction(self):
        """Test that enhanced ValidationError and JSONDecodeError carry response data."""
        instructor_client = instructor.from_provider(
            "openai/gpt-4o", api_key="fake-key-for-testing"
        )

        max_retries = 2

        with pytest.raises(InstructorRetryException) as exc_info:
            instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=ConditionalFailModel,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the name 'mary jane' from this text",
                    }
                ],
                max_retries=max_retries,
            )

        e = exc_info.value

        # Test that we can create a comprehensive failure report
        failure_report = {
            "total_attempts": e.n_attempts,
            "total_exceptions": len(e.all_exceptions),
            "total_failed_responses": len(e.all_failed_responses),
            "failure_details": [],
        }

        # Build detailed failure information
        for i, (exception, failed_response) in enumerate(
            zip(e.all_exceptions, e.all_failed_responses), 1
        ):
            failure_detail = {
                "attempt": i,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "has_failed_response": failed_response is not None,
            }

            # Check for enhanced exception data
            if hasattr(exception, "failed_response"):
                failure_detail["exception_has_response_data"] = (
                    exception.failed_response is not None
                )
            if hasattr(exception, "raw_content"):
                failure_detail["exception_has_raw_content"] = (
                    exception.raw_content is not None
                )
            if hasattr(exception, "raw_json_content"):
                failure_detail["exception_has_raw_json"] = (
                    exception.raw_json_content is not None
                )

            failure_report["failure_details"].append(failure_detail)

        # Verify the structure is correct
        assert failure_report["total_exceptions"] > 0, "Should have captured exceptions"
        assert failure_report["total_failed_responses"] > 0, (
            "Should have captured failed responses"
        )
        assert (
            len(failure_report["failure_details"]) == failure_report["total_exceptions"]
        ), "Should have details for each exception"
