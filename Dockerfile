FROM mcr.microsoft.com/powershell:7.4-ubuntu-22.04
ENV SYSTEM_MODULE="EntityCockpit" RUNTIME_AUTH="Robdoe" SYSTEM_ALIGN="12-CHANNEL"
COPY . /workspace
WORKDIR /workspace
CMD ["pwsh"]
