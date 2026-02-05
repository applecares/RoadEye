"""Species registry and definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ThreatStatus(str, Enum):
    """Threatened species status levels."""

    NONE = "none"
    VULNERABLE = "vulnerable"
    ENDANGERED = "endangered"
    CRITICALLY_ENDANGERED = "critically_endangered"


class SpeciesPriority(str, Enum):
    """Priority levels for species handling."""

    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    PEST = "pest"


@dataclass
class Species:
    """Species definition with metadata."""

    code: str
    scientific_name: str
    common_name: str
    mewc_class_id: Optional[int] = None
    threatened_status: ThreatStatus = ThreatStatus.NONE
    priority: SpeciesPriority = SpeciesPriority.NORMAL
    colour: str = "#868e96"  # Default gray

    @property
    def is_threatened(self) -> bool:
        """Check if species has threatened status."""
        return self.threatened_status != ThreatStatus.NONE

    @property
    def is_priority(self) -> bool:
        """Check if species requires priority handling."""
        return self.is_threatened or self.priority in (
            SpeciesPriority.CRITICAL,
            SpeciesPriority.HIGH,
            SpeciesPriority.PEST,
        )


@dataclass
class SpeciesRegistry:
    """Registry of species for a specific state/region."""

    state_code: str
    species: Dict[str, Species] = field(default_factory=dict)

    def register(self, species: Species) -> None:
        """Register a species."""
        self.species[species.code] = species

    def get(self, code: str) -> Optional[Species]:
        """Get species by code."""
        return self.species.get(code)

    def get_by_mewc_class(self, class_id: int) -> Optional[Species]:
        """Get species by MEWC class ID."""
        for species in self.species.values():
            if species.mewc_class_id == class_id:
                return species
        return None

    def list_threatened(self) -> List[Species]:
        """List all threatened species."""
        return [s for s in self.species.values() if s.is_threatened]

    def list_all(self) -> List[Species]:
        """List all registered species."""
        return list(self.species.values())

    def get_colour(self, code: str) -> str:
        """Get display colour for species."""
        species = self.get(code)
        return species.colour if species else "#868e96"

    @classmethod
    def from_config(cls, state_code: str, species_list: List[dict]) -> "SpeciesRegistry":
        """Create registry from configuration list."""
        registry = cls(state_code=state_code)

        for item in species_list:
            species = Species(
                code=item["code"],
                scientific_name=item["scientific_name"],
                common_name=item["common_name"],
                mewc_class_id=item.get("mewc_class_id"),
                threatened_status=ThreatStatus(item.get("threatened_status", "none")),
                priority=SpeciesPriority(item.get("priority", "normal")),
                colour=item.get("colour", "#868e96"),
            )
            registry.register(species)

        return registry


# Tasmania default species registry
def create_tasmania_registry() -> SpeciesRegistry:
    """Create the default Tasmania species registry."""
    registry = SpeciesRegistry(state_code="TAS")

    # Tasmanian species with MEWC class mappings
    species_data = [
        {
            "code": "DEVIL",
            "scientific_name": "Sarcophilus harrisii",
            "common_name": "Tasmanian Devil",
            "mewc_class_id": 0,
            "threatened_status": "endangered",
            "priority": "critical",
            "colour": "#ff6b6b",  # Red
        },
        {
            "code": "FCAT",
            "scientific_name": "Felis catus",
            "common_name": "Feral Cat",
            "mewc_class_id": 1,
            "threatened_status": "none",
            "priority": "pest",
            "colour": "#ffa94d",  # Orange
        },
        {
            "code": "PADEM",
            "scientific_name": "Thylogale billardierii",
            "common_name": "Tasmanian Pademelon",
            "mewc_class_id": 2,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#845ef7",  # Purple
        },
        {
            "code": "WALBY",
            "scientific_name": "Notamacropus rufogriseus",
            "common_name": "Bennett's Wallaby",
            "mewc_class_id": 3,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#5c7cfa",  # Blue
        },
        {
            "code": "WOMBAT",
            "scientific_name": "Vombatus ursinus",
            "common_name": "Bare-nosed Wombat",
            "mewc_class_id": 4,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#845ef7",  # Purple
        },
        {
            "code": "BPOSM",
            "scientific_name": "Trichosurus vulpecula",
            "common_name": "Brushtail Possum",
            "mewc_class_id": 5,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#22b8cf",  # Cyan
        },
        {
            "code": "FDEER",
            "scientific_name": "Dama dama",
            "common_name": "Fallow Deer",
            "mewc_class_id": 6,
            "threatened_status": "none",
            "priority": "pest",
            "colour": "#ffa94d",  # Orange
        },
        {
            "code": "BANDI",
            "scientific_name": "Isoodon obesulus",
            "common_name": "Southern Brown Bandicoot",
            "mewc_class_id": 7,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#74c0fc",  # Light blue
        },
        {
            "code": "CURRA",
            "scientific_name": "Strepera sp.",
            "common_name": "Currawong",
            "mewc_class_id": 8,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#51cf66",  # Green
        },
        {
            "code": "BRONZ",
            "scientific_name": "Phaps sp.",
            "common_name": "Bronzewing",
            "mewc_class_id": 9,
            "threatened_status": "none",
            "priority": "normal",
            "colour": "#51cf66",  # Green
        },
    ]

    for item in species_data:
        species = Species(
            code=item["code"],
            scientific_name=item["scientific_name"],
            common_name=item["common_name"],
            mewc_class_id=item["mewc_class_id"],
            threatened_status=ThreatStatus(item["threatened_status"]),
            priority=SpeciesPriority(item["priority"]),
            colour=item["colour"],
        )
        registry.register(species)

    return registry


# Global registry instance (lazy loaded)
_registry: Optional[SpeciesRegistry] = None


def get_species_registry() -> SpeciesRegistry:
    """Get the global species registry."""
    global _registry
    if _registry is None:
        _registry = create_tasmania_registry()
    return _registry


def configure_species_registry(registry: SpeciesRegistry) -> None:
    """Configure the global species registry."""
    global _registry
    _registry = registry
