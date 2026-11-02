# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Human-readable random names, e.g. for runs."""

from __future__ import annotations
import random
from collections.abc import Collection, Sequence
from chuchichaestli.utils.functools import prod


__all__ = [
    "QUALITIES",
    "COLORS",
    "SIZES",
    "ANIMALS",
    "CREATURES",
    "NATURE",
    "COSMOS",
    "ADJECTIVES",
    "NOUNS",
    "SlugGenerator",
    "generate_slug",
    "unique_name",
]


QUALITIES: tuple[str, ...] = (
    "able", "adamant", "adept", "agile", "airy", "amiable", "ample", "ancient",
    "arcane", "arctic", "ardent", "artful", "astral", "audacious", "awake", "balmy",
    "blazing", "blissful", "blithe", "blustery", "bold", "bouncy", "boundless",
    "brave", "brazen", "breezy", "bright", "brilliant", "brisk", "bubbly",
    "buoyant", "burly", "calm", "candid", "celestial", "cheerful", "chipper",
    "chromatic", "civil", "classic", "clever", "coastal", "cordial", "cosmic",
    "cozy", "crackling", "crafty", "crisp", "crystalline", "cunning", "curious",
    "dainty", "dapper", "daring", "dashing", "dauntless", "dazzling", "deep",
    "defiant", "deft", "devoted", "dewy", "distant", "doughty", "dreamy",
    "drifting", "dusky", "dynamic", "eager", "early", "earnest", "easy", "eclectic",
    "elated", "electric", "elegant", "eloquent", "elusive", "enchanted", "endless",
    "enigmatic", "epic", "eternal", "ethereal", "even", "exact", "exotic", "fabled",
    "fabulous", "fair", "famous", "fancy", "fast", "fearless", "feisty", "feral",
    "fervent", "festive", "fierce", "fiery", "fine", "firm", "flawless", "fleet",
    "fleeting", "flickering", "floral", "fluent", "fluffy", "flying", "fond",
    "fragrant", "frank", "free", "fresh", "frisky", "frosty", "frozen", "funky",
    "fuzzy", "galactic", "gallant", "gentle", "gifted", "gilded", "glacial", "glad",
    "gleaming", "glimmering", "glossy", "glowing", "graceful", "gracious", "grand",
    "gritty", "groovy", "gusty", "handy", "happy", "hardy", "harmonic", "hazy",
    "hearty", "heroic", "hidden", "honest", "hopeful", "howling", "humble",
    "humming", "hushed", "hypnotic", "icy", "ideal", "infinite", "intrepid",
    "iridescent", "jagged", "jaunty", "jazzy", "jolly", "jovial", "joyful",
    "jubilant", "keen", "kind", "kindly", "kinetic", "lavish", "legendary",
    "limber", "lively", "lofty", "loyal", "lucid", "lucky", "luminous", "lunar",
    "lush", "magic", "magnetic", "majestic", "major", "marvelous", "mellow",
    "mercurial", "merry", "misty", "modest", "molten", "mossy", "mysterious",
    "mystic", "mythic", "nautical", "neat", "nebulous", "nifty", "nimble", "noble",
    "nocturnal", "nomadic", "northern", "novel", "oaken", "optimal", "orbital",
    "ornate", "outlandish", "patient", "peaceful", "pearly", "peppy", "perky",
    "placid", "playful", "plucky", "poised", "polar", "polished", "primal", "prime",
    "prismatic", "proud", "quaint", "quantum", "quick", "quiet", "quirky",
    "radiant", "rapid", "rapt", "rare", "ready", "regal", "relentless", "resolute",
    "rich", "rising", "roaming", "roaring", "robust", "rocky", "rolling", "rosy",
    "rousing", "royal", "rugged", "rumbling", "rustic", "sandy", "savvy", "scenic",
    "scorching", "serene", "shadowy", "sharp", "shimmering", "shiny", "silent",
    "silken", "silvery", "simple", "sincere", "sizzling", "sleek", "sly", "smart",
    "smoldering", "smooth", "snappy", "snowy", "snug", "soaring", "solar", "solid",
    "sonic", "sonorous", "sparkling", "spectral", "spirited", "splendid", "spry",
    "stalwart", "starry", "steady", "stealthy", "stellar", "still", "stoic",
    "stormy", "strong", "sturdy", "sublime", "sunlit", "sunny", "super", "supreme",
    "swift", "swirling", "tactful", "tender", "thrifty", "thundering", "tidal",
    "tidy", "timeless", "tireless", "toasty", "tranquil", "tropical", "true",
    "trusty", "twilit", "twinkling", "unbound", "untamed", "upbeat", "urban",
    "valiant", "valorous", "velvety", "verdant", "vernal", "vibrant", "vigilant",
    "vivid", "volcanic", "wandering", "warm", "wavy", "whimsical", "whirling",
    "wild", "wily", "windy", "wintry", "wise", "wistful", "witty", "wondrous",
    "woven", "young", "zany", "zealous", "zesty", "zippy",
)  # fmt: skip

COLORS: tuple[str, ...] = (
    "alabaster", "amber", "amethyst", "auburn", "azure", "beige", "black", "blue",
    "bronze", "carmine", "celadon", "cerise", "cerulean", "charcoal", "chartreuse",
    "cinnabar", "citrine", "cobalt", "copper", "coral", "crimson", "cyan", "ebony",
    "emerald", "fuchsia", "garnet", "golden", "gray", "green", "indigo", "ivory",
    "jade", "jet", "khaki", "lavender", "lilac", "lime", "magenta", "mahogany",
    "maroon", "mauve", "mint", "navy", "obsidian", "ochre", "olive", "onyx", "opal",
    "orange", "peach", "periwinkle", "pewter", "pink", "platinum", "plum", "purple",
    "red", "ruby", "russet", "saffron", "sapphire", "scarlet", "sepia", "sienna",
    "silver", "slate", "tan", "tangerine", "tawny", "teal", "topaz", "turquoise",
    "ultramarine", "umber", "velvet", "vermilion", "violet", "viridian", "yellow",
)  # fmt: skip

SIZES: tuple[str, ...] = (
    "big", "colossal", "enormous", "gargantuan", "giant", "gigantic", "great",
    "huge", "hulking", "humongous", "immense", "jumbo", "large", "little",
    "mammoth", "massive", "mega", "micro", "mighty", "mini", "nano", "petite",
    "tall", "tiny", "titanic", "towering", "vast",
)  # fmt: skip

ANIMALS: tuple[str, ...] = (
    "aardvark", "albatross", "alpaca", "anaconda", "anteater", "antelope",
    "armadillo", "axolotl", "baboon", "badger", "barracuda", "bat", "beaver", "bee",
    "beetle", "bison", "bluejay", "boar", "bobcat", "buffalo", "bumblebee",
    "butterfly", "buzzard", "camel", "capybara", "caracal", "cardinal", "caribou",
    "cassowary", "cat", "catfish", "chameleon", "cheetah", "chickadee",
    "chinchilla", "chipmunk", "cicada", "coati", "cobra", "cockatoo", "condor",
    "cormorant", "cougar", "coyote", "crab", "crane", "cricket", "crocodile",
    "crow", "cuckoo", "cuttlefish", "deer", "dingo", "dolphin", "donkey",
    "dormouse", "dove", "dragonfly", "duck", "dugong", "eagle", "eel", "egret",
    "elephant", "elk", "emu", "ermine", "falcon", "ferret", "finch", "firefly",
    "flamingo", "fox", "frog", "gazelle", "gecko", "gerbil", "gharial", "gibbon",
    "giraffe", "gnu", "goat", "goldfinch", "goose", "gopher", "gorilla",
    "grasshopper", "grouse", "gull", "hamster", "hare", "harrier", "hawk",
    "hedgehog", "heron", "hippo", "hornbill", "hornet", "horse", "hummingbird",
    "husky", "hyena", "ibex", "ibis", "iguana", "impala", "jackal", "jackrabbit",
    "jaguar", "jay", "jellyfish", "kangaroo", "kestrel", "kingfisher", "kite",
    "kiwi", "koala", "koi", "kookaburra", "krill", "ladybug", "lark", "lemming",
    "lemur", "leopard", "limpet", "lion", "lionfish", "lizard", "llama", "lobster",
    "loon", "lynx", "macaw", "magpie", "mallard", "mamba", "manatee", "mandrill",
    "mantis", "marlin", "marmoset", "marmot", "marten", "meerkat", "merlin", "mink",
    "mole", "mongoose", "monkey", "moose", "moth", "mouse", "mule", "mustang",
    "narwhal", "nautilus", "newt", "nightingale", "ocelot", "octopus", "okapi",
    "opossum", "orangutan", "orca", "oriole", "osprey", "ostrich", "otter", "owl",
    "ox", "oyster", "panda", "pangolin", "panther", "parrot", "peacock", "pelican",
    "penguin", "peregrine", "pheasant", "pigeon", "pika", "piranha", "platypus",
    "plover", "pony", "porcupine", "possum", "pronghorn", "puffin", "puma",
    "python", "quail", "quetzal", "quokka", "rabbit", "raccoon", "ram", "raptor",
    "rattlesnake", "raven", "reindeer", "rhino", "robin", "sailfish", "salamander",
    "salmon", "sandpiper", "sardine", "scarab", "scorpion", "seahorse", "seal",
    "serval", "shark", "sheep", "shrew", "shrimp", "sidewinder", "skunk", "sloth",
    "snail", "snake", "sparrow", "spider", "squid", "squirrel", "stag", "stallion",
    "starling", "stingray", "stork", "swallow", "swan", "swordfish", "tamarin",
    "tapir", "tarantula", "tarsier", "tern", "tiger", "toad", "tortoise", "toucan",
    "trout", "tuna", "turkey", "turtle", "viper", "vole", "vulture", "wallaby",
    "walrus", "warbler", "wasp", "weasel", "whale", "wildcat", "wildebeest", "wolf",
    "wolverine", "wombat", "woodpecker", "wren", "yak", "zebra",
)  # fmt: skip

CREATURES: tuple[str, ...] = (
    "amphisbaena", "bahamut", "basilisk", "behemoth", "bunyip", "cerberus",
    "chimera", "cockatrice", "dragon", "drake", "fenrir", "firebird", "gargoyle",
    "garuda", "griffin", "hippocampus", "hippogriff", "hydra", "jackalope",
    "jormungandr", "kelpie", "kirin", "kitsune", "kraken", "leviathan", "manticore",
    "ouroboros", "pegasus", "phoenix", "roc", "simurgh", "sleipnir", "tarasque",
    "thunderbird", "unicorn", "wyrm", "wyvern",
)  # fmt: skip

NATURE: tuple[str, ...] = (
    "acorn", "alder", "anchor", "apple", "arch", "arrow", "aspen", "atlas",
    "autumn", "avalanche", "bamboo", "banner", "basil", "basin", "bay", "bayou",
    "beach", "beacon", "berry", "birch", "blizzard", "bloom", "blossom", "bonfire",
    "boulder", "breeze", "brick", "brook", "butte", "cactus", "cairn", "canopy",
    "canyon", "cape", "cascade", "cavern", "cedar", "cherry", "cinder", "citrus",
    "cliff", "cloud", "clover", "coast", "cove", "crag", "crater", "creek", "crest",
    "crystal", "current", "cyclone", "cypress", "daisy", "dawn", "dell", "delta",
    "desert", "dew", "dream", "drift", "dune", "dusk", "echo", "elm", "ember",
    "estuary", "feather", "fen", "fern", "field", "fjord", "flame", "flint",
    "flower", "fog", "forest", "fountain", "frost", "gale", "garden", "geyser",
    "glacier", "glade", "glen", "granite", "grotto", "grove", "gulch", "hail",
    "harbor", "haze", "hazel", "heath", "hill", "hollow", "horizon", "inlet",
    "iris", "island", "isle", "ivy", "jasmine", "jungle", "juniper", "knoll",
    "lagoon", "lake", "lantern", "laurel", "lava", "leaf", "ledge", "lemon",
    "lichen", "lily", "lotus", "magma", "mango", "mangrove", "maple", "marsh",
    "meadow", "mesa", "mirage", "mist", "monsoon", "moor", "moss", "mountain",
    "nectar", "night", "nimbus", "oak", "oasis", "ocean", "orchid", "palm", "peak",
    "pebble", "pepper", "pine", "plateau", "plume", "pond", "poppy", "prairie",
    "prism", "quarry", "quartz", "rain", "rainbow", "rapids", "ravine", "reef",
    "ridge", "rift", "river", "rock", "rose", "sage", "savanna", "sea", "sequoia",
    "shadow", "shell", "shore", "sierra", "sky", "spire", "spring", "spruce",
    "squall", "steppe", "stone", "storm", "stream", "summit", "sunset", "tempest",
    "thicket", "thistle", "thorn", "thunder", "tide", "timber", "tornado",
    "torrent", "trail", "tree", "tulip", "tundra", "typhoon", "vale", "valley",
    "vine", "volcano", "walnut", "waterfall", "wave", "whirlwind", "wildfire",
    "willow", "wind", "wood", "yarrow", "zephyr", "zinnia",
)  # fmt: skip

COSMOS: tuple[str, ...] = (
    "apogee", "asteroid", "aurora", "comet", "constellation", "corona", "cosmos",
    "eclipse", "equinox", "exoplanet", "galaxy", "lightyear", "magnetar",
    "meridian", "meteor", "moon", "moonbeam", "nadir", "nebula", "nova", "orbit",
    "parsec", "photon", "planet", "pulsar", "quark", "quasar", "satellite",
    "solstice", "star", "starburst", "stardust", "starlight", "sun", "sunspot",
    "supernova", "wormhole", "zenith", "zodiac",
)  # fmt: skip

ADJECTIVES: tuple[str, ...] = QUALITIES + COLORS + SIZES
NOUNS: tuple[str, ...] = ANIMALS + CREATURES + NATURE + COSMOS


class SlugGenerator:
    """Draw slugs of one word from each part."""

    def __init__(
        self,
        parts: Sequence[Sequence[str]] = (ADJECTIVES, NOUNS),
        rand: random.Random | None = None,
        attempts: int = 100,
    ):
        """Constructor.

        Args:
            parts: Word lists, one word drawn from each in order.
            rand: Random source; a `random.SystemRandom` when absent.
            attempts: Draws allowed before giving up on a slug without a
                repeated word.

        Raises:
            ValueError: If there is no part, a part is empty, or `attempts`
                is not positive.
        """
        if not parts or any(not part for part in parts):
            raise ValueError("A slug needs at least one part, and no empty one.")
        if attempts < 1:
            raise ValueError(
                f"A slug needs a positive number of attempts, got {attempts!r}."
            )
        self.parts = tuple(tuple(part) for part in parts)
        self.rand = rand if rand is not None else random.SystemRandom()
        self.attempts = attempts

    def __repr__(self) -> str:
        """Return a short description of the generator."""
        parts, slugs = len(self.parts), self.combinations
        return f"{type(self).__name__}({parts} parts, {slugs} slugs)"

    @property
    def combinations(self) -> int:
        """Number of slugs the parts can form."""
        return prod(len(part) for part in self.parts)

    def generate(self) -> list[str]:
        """Return one word from each part, none repeated.

        Raises:
            ValueError: If no draw within `attempts` avoids a repeated word.
        """
        for _ in range(self.attempts):
            words = [self.rand.choice(part) for part in self.parts]
            if len(set(words)) == len(words):
                return words
        raise ValueError(
            f"No slug without a repeated word in {self.attempts} draws; "
            "the parts share too many words."
        )

    def slug(self, sep: str = "-") -> str:
        """Return one slug, its words joined.

        Args:
            sep: What joins the words.
        """
        return sep.join(self.generate())


def generate_slug(sep: str = "-", rand: random.Random | None = None) -> str:
    """Return one slug of an adjective and a noun.

    Args:
        sep: What joins the words.
        rand: Random source; a `random.SystemRandom` when absent.
    """
    return SlugGenerator(rand=rand).slug(sep)


def unique_name(
    taken: Collection[str] = (),
    generator: SlugGenerator | None = None,
    attempts: int = 8,
    sep: str = "-",
) -> str:
    """Return a slug not in `taken`, numbered only after repeated clashes.

    Args:
        taken: Names already in use.
        generator: Draws the slugs; the adjective-noun default when absent.
        attempts: Draws before a clash is resolved by a numeric suffix.
        sep: What joins the words and the suffix.
    """
    generator = generator if generator is not None else SlugGenerator()
    used = set(taken)
    name = generator.slug(sep)
    for _ in range(attempts - 1):
        if name not in used:
            return name
        name = generator.slug(sep)
    if name not in used:
        return name
    index = 2
    while f"{name}{sep}{index}" in used:
        index += 1
    return f"{name}{sep}{index}"
