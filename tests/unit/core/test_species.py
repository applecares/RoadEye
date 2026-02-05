"""Tests for species registry and definitions."""


from roadkill.core.species import (
    Species,
    SpeciesPriority,
    SpeciesRegistry,
    ThreatStatus,
    create_tasmania_registry,
    get_species_registry,
)


class TestSpecies:
    """Tests for Species dataclass."""

    def test_create_species(self):
        """Test creating a species."""
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
        )
        assert species.code == "TEST"
        assert species.scientific_name == "Testus speciesus"
        assert species.common_name == "Test Species"
        assert species.threatened_status == ThreatStatus.NONE
        assert species.priority == SpeciesPriority.NORMAL

    def test_is_threatened_false_by_default(self):
        """Test that species is not threatened by default."""
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
        )
        assert species.is_threatened is False

    def test_is_threatened_when_endangered(self):
        """Test that endangered species is marked as threatened."""
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
            threatened_status=ThreatStatus.ENDANGERED,
        )
        assert species.is_threatened is True

    def test_is_priority_for_threatened_species(self):
        """Test that threatened species has priority."""
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
            threatened_status=ThreatStatus.ENDANGERED,
        )
        assert species.is_priority is True

    def test_is_priority_for_pest(self):
        """Test that pest species has priority."""
        species = Species(
            code="PEST",
            scientific_name="Pestus animalus",
            common_name="Pest Animal",
            priority=SpeciesPriority.PEST,
        )
        assert species.is_priority is True


class TestSpeciesRegistry:
    """Tests for SpeciesRegistry."""

    def test_register_and_get_species(self):
        """Test registering and retrieving a species."""
        registry = SpeciesRegistry(state_code="TEST")
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
        )
        registry.register(species)

        retrieved = registry.get("TEST")
        assert retrieved is not None
        assert retrieved.code == "TEST"

    def test_get_nonexistent_species(self):
        """Test getting a species that doesn't exist."""
        registry = SpeciesRegistry(state_code="TEST")
        assert registry.get("NONEXISTENT") is None

    def test_get_by_mewc_class(self):
        """Test getting species by MEWC class ID."""
        registry = SpeciesRegistry(state_code="TEST")
        species = Species(
            code="TEST",
            scientific_name="Testus speciesus",
            common_name="Test Species",
            mewc_class_id=42,
        )
        registry.register(species)

        retrieved = registry.get_by_mewc_class(42)
        assert retrieved is not None
        assert retrieved.code == "TEST"

    def test_list_threatened(self):
        """Test listing threatened species."""
        registry = SpeciesRegistry(state_code="TEST")

        normal_species = Species(
            code="NORMAL",
            scientific_name="Normalus speciesus",
            common_name="Normal Species",
        )
        threatened_species = Species(
            code="THREAT",
            scientific_name="Threatus speciesus",
            common_name="Threatened Species",
            threatened_status=ThreatStatus.ENDANGERED,
        )

        registry.register(normal_species)
        registry.register(threatened_species)

        threatened = registry.list_threatened()
        assert len(threatened) == 1
        assert threatened[0].code == "THREAT"

    def test_list_all(self):
        """Test listing all species."""
        registry = SpeciesRegistry(state_code="TEST")
        registry.register(
            Species(code="A", scientific_name="A", common_name="A")
        )
        registry.register(
            Species(code="B", scientific_name="B", common_name="B")
        )

        all_species = registry.list_all()
        assert len(all_species) == 2


class TestTasmaniaRegistry:
    """Tests for Tasmania species registry."""

    def test_create_tasmania_registry(self):
        """Test creating the Tasmania registry."""
        registry = create_tasmania_registry()
        assert registry.state_code == "TAS"
        assert len(registry.list_all()) == 10

    def test_tasmania_devil_is_endangered(self):
        """Test that Tasmanian Devil is marked as endangered."""
        registry = create_tasmania_registry()
        devil = registry.get("DEVIL")

        assert devil is not None
        assert devil.common_name == "Tasmanian Devil"
        assert devil.threatened_status == ThreatStatus.ENDANGERED
        assert devil.is_threatened is True
        assert devil.priority == SpeciesPriority.CRITICAL

    def test_tasmania_feral_cat_is_pest(self):
        """Test that feral cat is marked as pest."""
        registry = create_tasmania_registry()
        cat = registry.get("FCAT")

        assert cat is not None
        assert cat.common_name == "Feral Cat"
        assert cat.priority == SpeciesPriority.PEST

    def test_get_species_registry_returns_tasmania(self):
        """Test that default registry is Tasmania."""
        registry = get_species_registry()
        assert registry.state_code == "TAS"
        assert registry.get("DEVIL") is not None
