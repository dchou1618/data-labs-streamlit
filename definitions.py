from mongoengine import Document, EmbeddedDocument
from mongoengine.fields import (
    DateTimeField,
    DecimalField,
    ReferenceField,
    StringField,
    IntField,
)

class Users(Document):
    meta = {"collection": "users"}
    name = StringField()
    description = StringField()
    _id = IntField()
    price = DecimalField()
    weight = DecimalField()
