FROM mcr.microsoft.com/powershell:7.4-ubuntu-22.04
ENV ATTESTATION_MODULE="EntityCockpit" RUNTIME_AUTH="Robdoe" SYSTEM_STATE = "GLYPHIC_CRYSTALLIZED"
COPY . /workspace
WORKDIR /workspace
CMD ["pwsh"]
