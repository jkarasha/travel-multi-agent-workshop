# Plan: Private Cosmos + ACA-hosted backend (Option A)

Status: Draft, pending review
Branch: `langfuse` (drafted alongside Langfuse migration; networking work to be branched separately when picked up)
Date: 2026-05-01

## Context

Tenant Azure Policy enforces `publicNetworkAccess: Disabled` on Cosmos DB
accounts. The current workshop infra (`02_completed/infra/main.bicep`) only
provisions Cosmos + OpenAI + managed identity + role assignments — no compute,
no VNet, no private endpoint, no private DNS zone. The intended developer flow
("run Python + Angular on the laptop, hit Cosmos directly") cannot work under
the policy: with public access disabled and no private endpoint, the Cosmos
account is unreachable from anywhere.

App auth is already AAD via `DefaultAzureCredential` against `COSMOSDB_ENDPOINT`
(`disableLocalAuth: true`), so the migration is purely a networking + hosting
change.

## Goals

1. Cosmos reachable only via private endpoint, end-to-end consistent with the
   policy.
2. App runs on Azure Container Apps inside the same VNet, using the existing
   managed identity.
3. Frontend can stay on the laptop during workshop exercises (Angular dev
   server proxies to ACA's public ingress).
4. Seeding runs inside the VNet, not from a laptop.
5. `azd up` continues to be the one-shot setup command for attendees.

## Scope

Implement in `02_completed/` first as the reference, then port to
`01_exercises/`.

## Infra additions (`02_completed/infra/`)

New bicep modules under `shared/`:

1. **`network.bicep`** — VNet `vnet-${token}` with three subnets:
   - `snet-aca` /23, delegated to `Microsoft.App/environments` (ACA
     workload-profile requirement)
   - `snet-pe` /27, `privateEndpointNetworkPolicies: Disabled`
   - `snet-jobs` /26, for the seed job (can share with `snet-aca` if we use the
     same env; keeping separate keeps blast radius small)
2. **`privatedns.bicep`** — Private DNS zone
   `privatelink.documents.azure.com`, VNet link, `registrationEnabled: false`.
   Add `privatelink.openai.azure.com` and `privatelink.azurecr.io` in the same
   module for symmetry.
3. **`privateendpoint.bicep`** — generic PE module called once per service
   (Cosmos `Sql` group, AOAI `account` group, ACR `registry` group). Each binds
   to its private DNS zone via `privateDnsZoneGroups`.
4. **`containerregistry.bicep`** — ACR Premium (Premium is required for private
   endpoints). `publicNetworkAccess: Disabled`. AcrPull granted to the workshop
   MI.
5. **`containerapps.bicep`** —
   - Log Analytics workspace (ACA dependency)
   - ACA Environment with
     `vnetConfiguration.infrastructureSubnetId = snet-aca`, workload profiles
     enabled, `internal = false` so the backend has a public FQDN for the local
     Angular dev server
   - One Container App for the FastAPI backend (image from ACR, MI attached,
     env vars wired from outputs of cosmos/openai modules, ingress 8000, CORS
     for `http://localhost:4200`)
6. **`seedjob.bicep`** — ACA Job (manual trigger, runs once), same image + MI,
   command `python data/seed_data.py`. Triggered from the postprovision hook
   via `az containerapp job start`.

## Changes to existing bicep

- `cosmosdb.bicep`: explicitly set `publicNetworkAccess: 'Disabled'` (currently
  policy-injected — making it explicit makes the IaC honest about the deploy
  target). Output `id` for the PE module to consume.
- `openai.bicep`: same — set `publicNetworkAccess: 'Disabled'` explicitly,
  output `id`.
- `main.bicep`: wire the new modules in dependency order:
  `network` -> `privatedns` -> `cosmos`/`openai`/`acr` -> `privateendpoint`
  (x3) -> `containerapps` -> `seedjob`. Add new outputs: `ACR_LOGIN_SERVER`,
  `CONTAINER_APP_FQDN`, `SEED_JOB_NAME`.

## Application changes

- **`02_completed/python/Dockerfile`** (new) — `python:3.12-slim`, install from
  `requirements.txt`, copy app, `CMD uvicorn app.travel_agents_api:app
  --host 0.0.0.0 --port 8000`. `.dockerignore` excluding `venv/`,
  `data/*.json` (seeded separately), `__pycache__`.
- **`travel_agents_api.py`**: add a `/healthz` endpoint for ACA probes; ensure
  CORS origins are configurable via env var.
- **`seed_data.py`**: no code change needed — it already uses
  `DefaultAzureCredential` and `COSMOSDB_ENDPOINT`. It just needs to run inside
  the VNet, which the seed job handles.

## `azure.yaml` changes

- Add a `services:` section pointing `api` at `./python` with
  `host: containerapp` and `language: py` (azd handles the ACR build + deploy).
- Replace the postprovision `python data/seed_data.py` block with: build+push
  image (azd does this for `services`), then
  `az containerapp job start --name $SEED_JOB_NAME --resource-group $RG_NAME`
  and tail logs. Drop the local venv setup from postprovision — that's now
  only needed if someone wants to run the agents locally for debugging
  (Option C territory, future).
- Update generated `.env` to point `MCP_SERVER_BASE_URL` and any backend URL
  at `https://$CONTAINER_APP_FQDN` so the local Angular frontend reaches the
  deployed API.

## Frontend

- `01_exercises/frontend/proxy.conf.json` and
  `02_completed/frontend/proxy.conf.json`: switch the proxy target from
  `http://localhost:8000` to `${API_BASE_URL}` injected at `npm start` time.
  Add a tiny `proxy.conf.js` that reads the env var so attendees don't
  hand-edit the file. No production build changes needed for the workshop.

## Attendee experience

1. `azd auth login`
2. `azd up` — provisions VNet/PEs/Cosmos/AOAI/ACR/ACA, builds + pushes image,
   starts seed job, prints frontend command.
3. `cd frontend && API_BASE_URL=https://$CONTAINER_APP_FQDN npm start`.

No VPN, no jumpbox, no laptop-to-Cosmos traffic.

## Deliberate non-goals

- Hub-spoke / shared private DNS — workshop is self-contained per attendee.
- Front Door / WAF / custom domain on ACA — not workshop-relevant.
- Pinning ACA env to internal-only with a public Application Gateway in front
  — overkill; the policy is about Cosmos, not the API.
- Local hot-reload story — defer to a follow-up using `az containerapp exec`
  or Dev Box (Option C).

## Open questions for review

1. **ACA subnet size** — `/23` is 512 addresses and may collide with attendees
   who have constrained CIDR allocation policy. If the tenant restricts VNet
   sizes, set a budget (e.g., must fit in a `/24`) and switch to ACA
   Consumption-only with smaller subnets.
2. **AOAI policy** — confirm whether AOAI is also
   `publicNetworkAccess: Disabled` in the tenant. If yes, plan is unchanged.
   If no, the AOAI PE is optional; default is to include it for consistency.
3. **ACR Premium SKU cost** — Premium is the only SKU that supports private
   endpoints. ~$0.66/day per attendee. Acceptable for a workshop?
4. **Workshop cleanup** — `azd down` should tear everything down cleanly. PEs
   and DNS records can be sticky; add explicit `dependsOn` ordering and verify
   a full down/up cycle.

## Rollout

- Step 1: write & validate networking + PE bicep on a throwaway sub.
- Step 2: add Dockerfile + ACA module, deploy backend, verify API hits Cosmos.
- Step 3: add seed job, verify clean `azd up` from empty.
- Step 4: switch frontend proxy + update README / `Module-00.md`.
- Step 5: port to `01_exercises/`.

## Discovery notes (snapshot at draft time)

- `02_completed/infra/main.bicep` deploys: managed identity, Cosmos DB
  (serverless, NoSQL, vector + full-text), OpenAI (gpt-4.1-mini +
  text-embedding-3-small), role assignments. No compute, no networking.
- `02_completed/infra/shared/cosmosdb.bicep` does not set
  `publicNetworkAccess`; the Disabled state is policy-injected at deploy.
- `02_completed/python/src/app/services/azure_cosmos_db.py` constructs
  `CosmosClient(COSMOS_DB_URL, credential=DefaultAzureCredential())`. AAD
  only; no key path.
- `02_completed/python/src/app/travel_agents_api.py` is a FastAPI app served
  by `uvicorn` on `0.0.0.0:$port`.
- `02_completed/azure.yaml` has no `services:` block and runs
  `python data/seed_data.py` from the developer machine in `postprovision` —
  this is the call that breaks under the policy and must move into the VNet.
- No Dockerfile exists in the repo today.
