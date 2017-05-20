using EduNetsUrl;

function randurl()::AbstractString
	builder = Vector{String}(0);
	push!(builder, "http");
	if rand(Bool)
		push!(builder, "s");
	end
	push!(builder, "://");
	if rand(Bool)
		push!(builder, "www.");
	end
	push!(builder, randstring(rand(1:100)));
	for i in 1:rand(0:10)
		push!(builder, ".");
		push!(builder, randstring(rand(1:100)));
	end
	push!(builder, "/");
	if rand(Bool)
		push!(builder, randstring(rand(1:100)));
		for i in 1:rand(0:10)
			push!(builder, "/");
			push!(builder, randstring(rand(1:100)));
		end
		if rand(Bool)
			push!(builder, ".");
			push!(builder, randstring(rand(1:10)));
		end
	end
	if rand(Bool)
		push!(builder, "?");
		push!(builder, randstring(rand(1:30)));
		push!(builder, "=");
		push!(builder, randstring(rand(1:30)));
		for i in 1:rand(0:10)
			push!(builder, "&");
			push!(builder, randstring(rand(1:30)));
			push!(builder, "=");
			push!(builder, randstring(rand(1:30)));
		end
	end
	return join(builder)
end

function testModel(model::Model)::Bool
	urls::Vector{AbstractString} = [randurl() for i in 1:100]
	labels = rand(1:2, 100);
	dataset = EduNetsUrl.processDataset(urls, labels; featureCount = 1000)

	reference = EduNetsUrl.project!(model, dataset)[end, :];
	for i in 1:100
		if EduNetsUrl.project!(model, dataset)[end, :] != reference
			return false;
		end
	end
	return true;
end

