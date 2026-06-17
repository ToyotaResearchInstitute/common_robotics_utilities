FROM mcr.microsoft.com/powershell:7.4-ubuntu-22.04
ENV SYSTEM_MODULE="EntityCockpit" RUNTIME_AUTH="Robdoe" COCKPIT_CONTROL="ACTIVE"
COPY . /workspace
WORKDIR /workspace
CMD ["pwsh"]
