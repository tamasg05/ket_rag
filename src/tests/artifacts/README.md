# PDF regression artifacts

`audi_a8_2026_price_list.pdf` is the Audi A8 price-list fixture
used to test complex, text-based PDF tables. It exercises:

- several tables on one page;
- side-by-side tables;
- rotated column headings;
- visually merged product rows with multiple price lines;
- missing unruled edge columns; and
- spaces used as thousands separators.

SHA-256:
`e3bee2b41ea4a94a81496f2690a5e6cb67093e9bc5097f95a5e644d43da52451`

The five public Opel price lists are used to test a second PDF layout family:

- unruled leading columns;
- header rows located inside the detected table box;
- removal of spaces used as thousands separators in prices;
- associations among equipment levels, battery sizes, ranges, and prices; and
- side-by-side technical-dimension tables.

| Artifact | SHA-256 |
| --- | --- |
| `opel_HU_Astra_Electric.pdf` | `d6ad648c3b257e8f4f15422ecfa2bec0261ccda02e0b11ee914e1401b5215b4f` |
| `opel_HU_Combo__Electric_egyteru.pdf` | `803b4def08e2e027db3996995afd19dc03bdc63ec6c7a3d46dfc306eaefd0f40` |
| `opel_HU_Frontera_Electric.pdf` | `545935e5876fcae586b286a8f73fc97c4363c8337d5e2bece01d2153f6ceaacc` |
| `opel_HU_Mokka_Electric_MY25 1.pdf` | `9eb727bd60de74c6f403dede09fa4bc4cd4f5da215afd3f134b15ff798f75e5b` |
| `opel_HU_Zafira_Electric.pdf` | `a16707fd6c126183b409d933f47a58e1c834f03070c93d2b2c8ba1638f483acf` |

Five additional public Audi price lists are used to test long documents:

- repeated model, standard-equipment, and optional-equipment tables;
- a consistent eight-column model-price structure;
- representative petrol, diesel, plug-in-hybrid, and electric rows;
- price tables continued on a second page;
- removal of spaces used as thousands separators in prices; and
- conversion of table rows into the structured text supplied to RAG.

| Artifact | SHA-256 |
| --- | --- |
| `A3_Limousine.pdf` | `35e93ec09829cf58257df5044fd9554bc76ec31dbba3faf0cf3eb95a2847987b` |
| `A5_Avant.pdf` | `724c0fbec783a8f80df80b34ab760600fa06ba12cb2cdb1f40cb50df5561a8ac` |
| `A6_Limousine.pdf` | `2fb3ff2c374d88131a7e3e3353ef17dc6bc7e77838a471b95c73293a008dda60` |
| `Q4_Sportback_e-tron.pdf` | `c8073d976b9228d07a6ce5c2a7824e7a8c924b3b6428625e8388ae64fdae223b` |
| `Q8.pdf` | `0858e1fb90400f3d49d52da7061d06ef647634f7bd0add07d74db3a7415c5567` |
