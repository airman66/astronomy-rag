# Planets score 1.1 reference

`planets-score-1.1.json` contains 32 synthetic NASA-shaped input rows and 24 expected candidates. It is not a downloaded NASA snapshot.

Expected responses were computed with the original `Models.cs`, `Habitability.cs` and `RankingUncertainty.cs` in `services/planets-api/Planets.Api`, using .NET 10 and `System.Random(20261114)`. All candidate fields, flags, ranks and stability values are compared by `test_matches_original_dotnet_reference`. The Python test does not require .NET or network access.

Inputs cover the stellar-temperature range, conservative/optimistic/excluded zones, asymmetric errors, measurement limits, missing radius/eccentricity, minimum masses and derived flux.
