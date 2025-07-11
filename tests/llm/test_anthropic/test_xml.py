from typing import Literal

import anthropic
import pytest
from pydantic import BaseModel, field_validator

import instructor
from instructor.function_calls import OpenAISchema
from instructor.retry import InstructorRetryException


class User(BaseModel):
    name: str
    age: int


client = instructor.from_anthropic(
    anthropic.Anthropic(), mode=instructor.Mode.ANTHROPIC_XML
)


def test_simple_xml():
    class User(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_is_uppercase(cls, v: str):
            assert v.isupper(), (
                f"{v} is not an uppercased string. Note that all characters in {v} must be uppercase (EG. TIM SARAH ADAM)."
            )
            return v

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        max_retries=2,
        system="Make sure to follow the instructions carefully and return a response object that matches the xml schema requested. Age is an integer.",
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old.",
            },
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "JOHN"  # due to validation
    assert resp.age == 18


def test_nested_type_xml():
    class Address(BaseModel):
        house_number: int
        street_name: str

    class User(BaseModel):
        name: str
        age: int
        address: Address

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old and lives at 123 First Avenue.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "John"
    assert resp.age == 18

    assert isinstance(resp.address, Address)
    assert resp.address.house_number == 123
    assert resp.address.street_name == "First Avenue"


def test_list_str_xml():
    class User(BaseModel):
        name: str
        age: int
        family: list[str]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        system="Make sure to follow the instructions carefully and return a response object that matches the xml schema requested. Family members here is just asking for a list of names",
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and family members.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert isinstance(resp.family, list)
    for member in resp.family:
        assert isinstance(member, str)


def test_literal_xml():
    class User(BaseModel):
        name: str
        role: Literal["admin", "user"]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": "Create a admin user for a model with a name and role.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.role == "admin"


def test_nested_list_xml():
    class Properties(BaseModel):
        key: str
        value: str

    class User(BaseModel):
        name: str
        age: int
        properties: list[Properties]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    for property in resp.properties:
        assert isinstance(property, Properties)


def test_system_messages_allcaps_xml():
    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "system",
                "content": "Please make sure to follow the instructions carefully and return a valid response object. All strings must be fully capitalised in all caps. (Eg. THIS IS AN UPPERCASE STRING) and age is an integer.",
            },
            {
                "role": "user",
                "content": "Create a user for a model with a name and age.",
            },
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name.isupper()


def test_retry_error_xml():
    class User(BaseModel):
        name: str

        @field_validator("name")
        def validate_name(cls, _):
            raise ValueError("Never succeed")

    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": "Extract John is 18 years old",
                },
            ],
            response_model=User,
        )
    except InstructorRetryException as e:
        assert e.total_usage > 0


@pytest.mark.asyncio
async def test_async_retry_error_xml():
    class User(BaseModel):
        name: str

        @field_validator("name")
        def validate_name(cls, _):
            raise ValueError("Never succeed")

    aclient = instructor.from_anthropic(
        anthropic.AsyncAnthropic(), mode=instructor.Mode.ANTHROPIC_XML
    )
    try:
        await aclient.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": "Extract John is 18 years old",
                },
            ],
            response_model=User,
        )
    except InstructorRetryException as e:
        assert e.total_usage > 0


def test_xml_schema_generation():
    class Address(OpenAISchema):
        house_number: int
        street_name: str

    class User(OpenAISchema):
        name: str
        age: int
        address: Address

    xml_schema = User.xml_schema

    assert "<User>" in xml_schema
    assert "</User>" in xml_schema
    assert "<name>" in xml_schema
    assert "<age>" in xml_schema
    assert "<address>" in xml_schema
    assert "<house_number>" in xml_schema
    assert "<street_name>" in xml_schema


def test_complex_nested_xml():
    class Contact(BaseModel):
        email: str
        phone: str

    class Address(BaseModel):
        street: str
        city: str
        zipcode: str

    class Person(BaseModel):
        name: str
        age: int
        contacts: list[Contact]
        address: Address
        is_active: bool

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2048,
        max_retries=1,
        messages=[
            {
                "role": "user",
                "content": "Create a person named Alice, age 30, with email alice@example.com, phone 555-1234, living at 123 Main St, New York, 10001, and is active.",
            }
        ],
        response_model=Person,
    )

    assert isinstance(resp, Person)
    assert resp.name == "Alice"
    assert resp.age == 30
    assert resp.is_active is True
    assert isinstance(resp.contacts, list)
    assert len(resp.contacts) > 0
    assert isinstance(resp.address, Address)


@pytest.mark.asyncio
async def test_async_xml():
    aclient = instructor.from_anthropic(
        anthropic.AsyncAnthropic(), mode=instructor.Mode.ANTHROPIC_XML
    )
    resp = await aclient.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract Sarah is 25 years old.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "Sarah"
    assert resp.age == 25
