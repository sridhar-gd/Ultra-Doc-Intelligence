# backend/schemas/shipment_schema.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class StopLocation(BaseModel):
    """A pickup or delivery stop with address details."""

    name: Optional[str] = Field(None, description="Facility or company name at this stop")
    address: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State / province code (e.g. CA)")
    zip_code: Optional[str] = Field(None, description="ZIP or postal code")
    country: Optional[str] = Field(default="USA", description="Country")
    appointment_time: Optional[datetime] = Field(
        None, description="Scheduled appointment datetime at this stop"
    )
    special_instructions: Optional[str] = Field(
        None, description="Stop-specific instructions (e.g. dock #, lumper required)"
    )


class DriverInfo(BaseModel):
    """Driver and equipment details from the carrier RC or BOL."""

    driver_name: Optional[str] = Field(None, description="Driver's full name")
    driver_phone: Optional[str] = Field(None, description="Driver's contact phone number")
    truck_number: Optional[str] = Field(None, description="Truck / tractor unit number")
    trailer_number: Optional[str] = Field(None, description="Trailer unit number")


class ShipmentData(BaseModel):
    """
    Structured extraction output for a logistics shipment document.

    Used as PydanticAI ExtractionAgent output_type:
        agent = Agent(
            model="anthropic:claude-sonnet-4-20250514",
            output_type=ShipmentData,
        )

    Claude is instructed to populate every field it can find in the document
    and set missing fields to null.  PydanticAI validates and retries if
    the output doesn't conform to this schema.
    """

    # Core identifiers
    shipment_id: Optional[str] = Field(
        None,
        description="Load / shipment ID (e.g. LD53657)",
    )
    pro_number: Optional[str] = Field(
        None,
        description="PRO number or BOL number",
    )
    reference_numbers: Optional[list[str]] = Field(
        default_factory=list,
        description="Additional reference numbers (PO, SO, customer ref, etc.)",
    )

    # Parties
    shipper: Optional[str] = Field(
        None,
        description="Shipper / consignor name and location",
    )
    consignee: Optional[str] = Field(
        None,
        description="Consignee / receiver name and location",
    )
    carrier_name: Optional[str] = Field(
        None,
        description="Carrier company name (e.g. SWIFT SHIFT LOGISTICS LLC)",
    )
    carrier_mc_number: Optional[str] = Field(
        None,
        description="Carrier MC (Motor Carrier) number",
    )
    broker_name: Optional[str] = Field(
        None,
        description="Freight broker / 3PL name if applicable",
    )

    # Stop details (structured)
    pickup: Optional[StopLocation] = Field(
        None,
        description="Pickup / origin stop details",
    )
    delivery: Optional[StopLocation] = Field(
        None,
        description="Delivery / destination stop details",
    )

    # Flat datetime fields (convenience duplicates of stop datetimes)
    pickup_datetime: Optional[datetime] = Field(
        None,
        description="Pickup appointment datetime (ISO 8601)",
    )
    delivery_datetime: Optional[datetime] = Field(
        None,
        description="Delivery appointment datetime (ISO 8601)",
    )

    # Equipment & mode
    equipment_type: Optional[str] = Field(
        None,
        description="Equipment type (e.g. Flatbed, Dry Van, Reefer, Step Deck)",
    )
    mode: Optional[str] = Field(
        None,
        description="Shipment mode (e.g. FTL, LTL, Intermodal)",
    )

    # Rate & financials
    rate: Optional[float] = Field(
        None,
        description="Carrier pay rate (numeric value only)",
    )
    currency: Optional[str] = Field(
        default="USD",
        description="Currency code (e.g. USD, CAD)",
    )
    rate_type: Optional[str] = Field(
        None,
        description="Rate type: Flat, Per Mile, Per CWT, etc.",
    )
    fuel_surcharge: Optional[float] = Field(
        None,
        description="Fuel surcharge amount if itemised",
    )
    total_charges: Optional[float] = Field(
        None,
        description="Total all-in charges (rate + surcharges)",
    )

    # Commodity & freight
    commodity: Optional[str] = Field(
        None,
        description="Description of goods / commodity",
    )
    weight: Optional[str] = Field(
        None,
        description="Total shipment weight including units (e.g. 56000 lbs)",
    )
    pieces: Optional[int] = Field(
        None,
        description="Number of pieces / pallets / units",
    )
    hazmat: Optional[bool] = Field(
        default=False,
        description="True if shipment contains hazardous materials",
    )
    temperature_requirement: Optional[str] = Field(
        None,
        description="Temperature range for reefer loads (e.g. 34-38°F)",
    )

    # Driver info
    driver: Optional[DriverInfo] = Field(
        None,
        description="Driver and equipment details",
    )

    # Special instructions
    special_instructions: Optional[str] = Field(
        None,
        description="General shipment-level special instructions or notes",
    )

    # Source metadata - Populated for Extracttion service
    document_id: Optional[str] = Field(
        None,
        description="UUID of the source document this was extracted from",
    )
    extraction_confidence: Optional[float] = Field(
        None,
        description="Overall confidence in this extraction (0-1), set by the service layer",
    )